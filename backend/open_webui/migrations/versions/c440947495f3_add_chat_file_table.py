"""Add chat_file table

Revision ID: c440947495f3
Revises: 81cc2ce44d79
Create Date: 2025-12-21 20:27:41.694897

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c440947495f3'
down_revision: Union[str, None] = '81cc2ce44d79'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    # PostgreSQL requires a PK or UNIQUE constraint on the referenced column for FK.
    # Old databases may have 'chat.id' without either, so ensure uniqueness first.
    for table_name in ('chat', 'file'):
        if table_name not in inspector.get_table_names():
            continue
        pk_cols = inspector.get_pk_constraint(table_name).get('constrained_columns', [])
        unique_cols = {tuple(u['column_names']) for u in inspector.get_unique_constraints(table_name)}
        if 'id' not in pk_cols and ('id',) not in unique_cols:
            with op.batch_alter_table(table_name) as batch_op:
                batch_op.create_unique_constraint(f'uq_{table_name}_id', ['id'])

    if 'chat_file' not in inspector.get_table_names():
        op.create_table(
            'chat_file',
            sa.Column('id', sa.Text(), primary_key=True),
            sa.Column('user_id', sa.Text(), nullable=False),
            sa.Column(
                'chat_id',
                sa.Text(),
                sa.ForeignKey('chat.id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column(
                'file_id',
                sa.Text(),
                sa.ForeignKey('file.id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column('message_id', sa.Text(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
            sa.Index('ix_chat_file_chat_id', 'chat_id'),
            sa.Index('ix_chat_file_file_id', 'file_id'),
            sa.Index('ix_chat_file_message_id', 'message_id'),
            sa.Index('ix_chat_file_user_id', 'user_id'),
            sa.UniqueConstraint('chat_id', 'file_id', name='uq_chat_file_chat_file'),
        )


def downgrade() -> None:
    op.drop_table('chat_file')
    pass
