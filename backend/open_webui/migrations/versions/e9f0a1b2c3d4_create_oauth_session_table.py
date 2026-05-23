"""Create oauth_session table (idempotent)

Revision ID: e9f0a1b2c3d4
Revises: a0b1c2d3e4f5
Create Date: 2026-05-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'e9f0a1b2c3d4'
down_revision: Union[str, None] = 'a0b1c2d3e4f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing_tables = inspector.get_table_names()

    if 'oauth_session' not in existing_tables:
        op.create_table(
            'oauth_session',
            sa.Column('id', sa.Text(), primary_key=True, nullable=False, unique=True),
            sa.Column(
                'user_id',
                sa.Text(),
                sa.ForeignKey('user.id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column('provider', sa.Text(), nullable=False),
            sa.Column('token', sa.Text(), nullable=False),
            sa.Column('expires_at', sa.BigInteger(), nullable=False),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )

    existing_indexes = {idx['name'] for idx in inspector.get_indexes('oauth_session')} if 'oauth_session' in inspector.get_table_names() else set()
    if 'idx_oauth_session_user_id' not in existing_indexes:
        op.create_index('idx_oauth_session_user_id', 'oauth_session', ['user_id'])
    if 'idx_oauth_session_expires_at' not in existing_indexes:
        op.create_index('idx_oauth_session_expires_at', 'oauth_session', ['expires_at'])
    if 'idx_oauth_session_user_provider' not in existing_indexes:
        op.create_index('idx_oauth_session_user_provider', 'oauth_session', ['user_id', 'provider'])


def downgrade() -> None:
    op.drop_index('idx_oauth_session_user_provider', table_name='oauth_session')
    op.drop_index('idx_oauth_session_expires_at', table_name='oauth_session')
    op.drop_index('idx_oauth_session_user_id', table_name='oauth_session')
    op.drop_table('oauth_session')
