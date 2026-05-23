"""Add oauth_session table

Revision ID: 38d63c18f30f
Revises: 3af16a1c9fb6
Create Date: 2025-09-08 14:19:59.583921

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '38d63c18f30f'
down_revision: Union[str, None] = '3af16a1c9fb6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Logic moved to a new idempotent migration (fix_oauth_session_idempotent).
    # This stub remains so the revision chain stays intact.
    pass


def downgrade() -> None:
    pass
