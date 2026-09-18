"""Add chunk_count to documents

Revision ID: 6d5529837bca
Revises: b71a4f2e9c3d
Create Date: 2026-08-30 20:05:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d5529837bca'
down_revision: Union[str, None] = 'b71a4f2e9c3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('documents', 'chunk_count')
