#!/usr/bin/env bash
# Create a Vultr instance sized for OmniMind and print its IP.
#
#   VULTR_API_KEY=... ./vultr_provision.sh [label] [region] [plan]
#
# Only provisions the server. deploy.sh then installs and starts the stack on it,
# so this stays provider-specific and that stays provider-agnostic.

set -euo pipefail

API="${VULTR_API_KEY:?set VULTR_API_KEY (https://my.vultr.com/settings/#settingsapi)}"
LABEL="${1:-omnimind}"
REGION="${2:-blr}"          # Bangalore — closest to the user; lowest RTT to the browser
# vc2-1c-2gb: 1 vCPU / 2GB RAM / 55GB SSD, ~$10/mo.
# Sizing rationale, from measurement rather than guesswork: the service idles at
# 256MB and peaks under ~700MB while ingesting a document. 1GB would run but leaves
# no headroom to BUILD the ~3.7GB image (pip resolving torch is memory-hungry), and
# an OOM-killed build is a confusing failure. 2GB builds and runs comfortably.
PLAN="${3:-vc2-1c-2gb}"
OS_ID=2284                   # Ubuntu 24.04 LTS x64

v() { curl -sS -H "Authorization: Bearer ${API}" -H "Content-Type: application/json" "$@"; }

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

say "Verifying API key"
if ! v https://api.vultr.com/v2/account | grep -q '"account"'; then
    echo "API key rejected. Note Vultr's API has an IP allow-list:"
    echo "  https://my.vultr.com/settings/#settingsapi — add this machine's IP."
    v https://api.vultr.com/v2/account | head -5
    exit 1
fi

say "Registering SSH key"
PUBKEY="$(cat ~/.ssh/id_rsa.pub)"
# Reuse an existing key with the same material rather than adding a duplicate on
# every run, which would clutter the account and make the right key ambiguous.
KEY_ID=$(v https://api.vultr.com/v2/ssh-keys | python -c "
import sys, json
want = '''${PUBKEY}'''.strip().split()[1]
for k in json.load(sys.stdin).get('ssh_keys', []):
    if k['ssh_key'].strip().split()[1] == want:
        print(k['id']); break
" 2>/dev/null || true)

if [ -z "${KEY_ID}" ]; then
    KEY_ID=$(v -X POST https://api.vultr.com/v2/ssh-keys \
        -d "{\"name\":\"${LABEL}-key\",\"ssh_key\":\"${PUBKEY}\"}" \
        | python -c "import sys,json; print(json.load(sys.stdin)['ssh_key']['id'])")
    echo "uploaded new key ${KEY_ID}"
else
    echo "reusing existing key ${KEY_ID}"
fi

say "Creating ${PLAN} in ${REGION}"
EXISTING=$(v https://api.vultr.com/v2/instances | python -c "
import sys, json
for i in json.load(sys.stdin).get('instances', []):
    if i.get('label') == '${LABEL}':
        print(i['id'], i.get('main_ip','')); break
" 2>/dev/null || true)

if [ -n "${EXISTING}" ]; then
    INSTANCE_ID=$(echo "$EXISTING" | awk '{print $1}')
    echo "instance '${LABEL}' already exists (${INSTANCE_ID}) — reusing"
else
    INSTANCE_ID=$(v -X POST https://api.vultr.com/v2/instances -d "{
        \"region\": \"${REGION}\",
        \"plan\": \"${PLAN}\",
        \"os_id\": ${OS_ID},
        \"label\": \"${LABEL}\",
        \"hostname\": \"${LABEL}\",
        \"sshkey_id\": [\"${KEY_ID}\"],
        \"backups\": \"disabled\",
        \"enable_ipv6\": false
    }" | python -c "import sys,json; print(json.load(sys.stdin)['instance']['id'])")
    echo "created ${INSTANCE_ID}"
fi

say "Waiting for the instance to become active"
for _ in $(seq 1 60); do
    read -r STATUS IP <<<"$(v "https://api.vultr.com/v2/instances/${INSTANCE_ID}" | python -c "
import sys, json
i = json.load(sys.stdin)['instance']
print(i.get('server_status','none'), i.get('main_ip','0.0.0.0'))")"
    [ "${STATUS}" = "ok" ] && [ "${IP}" != "0.0.0.0" ] && break
    sleep 10
done

if [ "${IP:-0.0.0.0}" = "0.0.0.0" ]; then
    echo "instance did not report an IP in time; check https://my.vultr.com/"
    exit 1
fi

say "Waiting for SSH"
for _ in $(seq 1 60); do
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
        "root@${IP}" true 2>/dev/null && break
    sleep 10
done

echo
echo "IP:      ${IP}"
# sslip.io resolves <ip>.sslip.io to that IP, which gives Let's Encrypt a real
# hostname to issue against without owning a domain. This matters beyond
# tidiness: voice mode calls getUserMedia for the microphone, and browsers only
# expose that in a secure context — over plain http:// on a bare IP, voice is
# silently unavailable.
echo "Domain:  ${IP}.sslip.io"
echo
echo "Next:  ./deploy.sh ${IP} ${IP}.sslip.io <your-email>"
