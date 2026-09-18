"""Provision an Oracle Cloud Always Free ARM instance for OmniMind.

Unlike most VPS providers, OCI gives you no network by default — an instance with no
VCN, gateway, route and security rule behind it is simply unreachable. So this builds
the whole path: VCN -> internet gateway -> route table -> security list -> public
subnet -> instance.

Idempotent by display name throughout, so re-running adopts what already exists rather
than creating duplicates (OCI happily makes five identically-named VCNs otherwise).

Sizing is pinned to the Always Free envelope — 4 OCPU / 24GB of VM.Standard.A1.Flex —
NOT to the tenancy's current limits. A trial tenancy reports far higher limits (41
OCPU / 277GB here), and anything provisioned above the free envelope stops being free
when the 30-day trial ends: it gets billed or reclaimed. Staying inside the envelope
means this instance keeps running for free indefinitely.
"""

from __future__ import annotations

import os
import sys
import time

import oci

APP = "omnimind"
OCPUS = 4          # Always Free ceiling for A1.Flex
MEMORY_GB = 24     # Always Free ceiling for A1.Flex
IMAGE_OCID = (
    "ocid1.image.oc1.ap-hyderabad-1."
    "aaaaaaaaiqjw7o5u3ecja5rayhq3jzhkut4werr6qcautsheuqbwrm7slhga"  # Ubuntu 24.04 aarch64
)
VCN_CIDR = "10.0.0.0/16"
SUBNET_CIDR = "10.0.1.0/24"


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def main() -> int:
    cfg = oci.config.from_file(os.path.expanduser("~/.oci/config"), "DEFAULT")
    tenancy = cfg["tenancy"]
    net = oci.core.VirtualNetworkClient(cfg)
    compute = oci.core.ComputeClient(cfg)
    iam = oci.identity.IdentityClient(cfg)

    ad = iam.list_availability_domains(compartment_id=tenancy).data[0].name
    log(f"availability domain: {ad}")

    pubkey = open(os.path.expanduser("~/.ssh/id_rsa.pub")).read().strip()

    # ── VCN ───────────────────────────────────────────────────
    vcn = next((v for v in net.list_vcns(compartment_id=tenancy).data
                if v.display_name == f"{APP}-vcn" and v.lifecycle_state == "AVAILABLE"), None)
    if vcn is None:
        log("creating VCN")
        vcn = net.create_vcn(oci.core.models.CreateVcnDetails(
            compartment_id=tenancy, cidr_block=VCN_CIDR,
            display_name=f"{APP}-vcn", dns_label=APP,
        )).data
        oci.wait_until(net, net.get_vcn(vcn.id), "lifecycle_state", "AVAILABLE", max_wait_seconds=300)
    else:
        log("reusing existing VCN")

    # ── Internet gateway ──────────────────────────────────────
    igw = next((g for g in net.list_internet_gateways(compartment_id=tenancy, vcn_id=vcn.id).data
                if g.lifecycle_state == "AVAILABLE"), None)
    if igw is None:
        log("creating internet gateway")
        igw = net.create_internet_gateway(oci.core.models.CreateInternetGatewayDetails(
            compartment_id=tenancy, vcn_id=vcn.id, is_enabled=True,
            display_name=f"{APP}-igw",
        )).data
        oci.wait_until(net, net.get_internet_gateway(igw.id), "lifecycle_state", "AVAILABLE",
                       max_wait_seconds=300)
    else:
        log("reusing internet gateway")

    # ── Default route out ─────────────────────────────────────
    # Without this the instance gets a public IP that nothing can reach.
    rt = net.get_route_table(vcn.default_route_table_id).data
    if not any(r.network_entity_id == igw.id for r in rt.route_rules):
        log("adding default route to internet gateway")
        net.update_route_table(rt.id, oci.core.models.UpdateRouteTableDetails(route_rules=[
            oci.core.models.RouteRule(destination="0.0.0.0/0",
                                      destination_type="CIDR_BLOCK",
                                      network_entity_id=igw.id),
        ]))

    # ── Security list ─────────────────────────────────────────
    # OCI blocks inbound by default. This is the step people miss: the instance boots,
    # SSH may work via some images' defaults, but 80/443 silently never answer.
    log("setting ingress rules (22, 80, 443)")
    def tcp(port: int) -> oci.core.models.IngressSecurityRule:
        return oci.core.models.IngressSecurityRule(
            protocol="6", source="0.0.0.0/0", is_stateless=False,
            tcp_options=oci.core.models.TcpOptions(
                destination_port_range=oci.core.models.PortRange(min=port, max=port)),
        )

    net.update_security_list(
        vcn.default_security_list_id,
        oci.core.models.UpdateSecurityListDetails(
            ingress_security_rules=[tcp(22), tcp(80), tcp(443)],
            egress_security_rules=[oci.core.models.EgressSecurityRule(
                protocol="all", destination="0.0.0.0/0", is_stateless=False)],
        ),
    )

    # ── Subnet ────────────────────────────────────────────────
    subnet = next((s for s in net.list_subnets(compartment_id=tenancy, vcn_id=vcn.id).data
                   if s.display_name == f"{APP}-subnet" and s.lifecycle_state == "AVAILABLE"), None)
    if subnet is None:
        log("creating public subnet")
        subnet = net.create_subnet(oci.core.models.CreateSubnetDetails(
            compartment_id=tenancy, vcn_id=vcn.id, cidr_block=SUBNET_CIDR,
            display_name=f"{APP}-subnet", dns_label="public",
            prohibit_public_ip_on_vnic=False,
        )).data
        oci.wait_until(net, net.get_subnet(subnet.id), "lifecycle_state", "AVAILABLE",
                       max_wait_seconds=300)
    else:
        log("reusing subnet")

    # ── Instance ──────────────────────────────────────────────
    existing = next((i for i in compute.list_instances(compartment_id=tenancy).data
                     if i.display_name == APP
                     and i.lifecycle_state in ("RUNNING", "PROVISIONING", "STARTING")), None)

    if existing:
        log(f"instance already exists ({existing.lifecycle_state})")
        instance = existing
    else:
        # A1.Flex capacity in Always Free regions is contended and frees up in bursts.
        # Two strategies stacked: ask for progressively less (a smaller request fits a
        # smaller gap), and keep retrying, because the answer genuinely changes minute
        # to minute. The smallest rung still comfortably exceeds what this app needs —
        # measured at 256MB idle and 792MB peak — so dropping to it costs nothing real.
        ladder = [(OCPUS, MEMORY_GB), (2, 12), (1, 6)]
        deadline = time.time() + float(os.environ.get("OCI_CAPACITY_WAIT_SECONDS", 1800))
        instance = None
        attempt = 0

        while instance is None:
            attempt += 1
            for ocpus, mem in ladder:
                try:
                    log(f"attempt {attempt}: launching {ocpus} OCPU / {mem}GB A1.Flex")
                    instance = compute.launch_instance(oci.core.models.LaunchInstanceDetails(
                        compartment_id=tenancy,
                        availability_domain=ad,
                        display_name=APP,
                        shape="VM.Standard.A1.Flex",
                        shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
                            ocpus=ocpus, memory_in_gbs=mem),
                        source_details=oci.core.models.InstanceSourceViaImageDetails(
                            image_id=IMAGE_OCID, boot_volume_size_in_gbs=100),
                        create_vnic_details=oci.core.models.CreateVnicDetails(
                            subnet_id=subnet.id, assign_public_ip=True),
                        metadata={"ssh_authorized_keys": pubkey},
                    )).data
                    log(f"accepted at {ocpus} OCPU / {mem}GB")
                    break
                except oci.exceptions.ServiceError as e:
                    if e.status in (500, 429) and "capacity" in str(e.message).lower():
                        log(f"  no capacity at {ocpus}/{mem}")
                        continue
                    raise

            if instance is None:
                if time.time() > deadline:
                    log("still no A1.Flex capacity after retrying.")
                    log("This is Oracle's constraint, not a misconfiguration — the network")
                    log("stack is built and ready, so re-running this script is all that is")
                    log("needed once capacity appears. Raise OCI_CAPACITY_WAIT_SECONDS to")
                    log("keep trying for longer.")
                    return 2
                time.sleep(60)

    log("waiting for RUNNING")
    instance = oci.wait_until(compute, compute.get_instance(instance.id),
                              "lifecycle_state", "RUNNING", max_wait_seconds=900).data

    vnics = compute.list_vnic_attachments(compartment_id=tenancy, instance_id=instance.id).data
    ip = None
    for va in vnics:
        v = net.get_vnic(va.vnic_id).data
        if v.public_ip:
            ip = v.public_ip
            break

    if not ip:
        log("instance is running but has no public IP")
        return 1

    print()
    print(f"  Instance : {APP} ({OCPUS} OCPU / {MEMORY_GB}GB ARM)")
    print(f"  Public IP: {ip}")
    # sslip.io resolves <ip>.sslip.io to that IP, giving Let's Encrypt a real hostname
    # to issue against without owning a domain. Not cosmetic: voice mode calls
    # getUserMedia, which browsers only expose over HTTPS.
    print(f"  Domain   : {ip}.sslip.io")
    print()
    print(f"  Next: bash deploy/deploy.sh {ip} {ip}.sslip.io <your-email> ubuntu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
