from alembic import op
import sqlalchemy as sa

revision = "08a60a287c2e"
down_revision = "4ace53fd72c8"
branch_labels = None
depends_on = None

def upgrade():
    # Add 'embed' column to 'knowledge' table
    op.add_column(
        "knowledge",
        sa.Column("embed", sa.Boolean(), nullable=False, default=True),
    )

def downgrade():
    op.drop_column("knowledge", "embed")