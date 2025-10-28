"""create acca_tickets and acca_legs tables

Revision ID: 802e2b5296ef
Revises: 16320a3a8f82
Create Date: 2025-10-24 21:24:14.542248
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = "802e2b5296ef"
down_revision = "16320a3a8f82"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- acca_tickets table ---
    op.create_table(
        "acca_tickets",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("sport", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("note", sa.String(), nullable=True),
        sa.Column("stake_units", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("combined_price", sa.Float(), nullable=True),
        sa.Column("est_edge", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, default=datetime.utcnow),
    )

    # --- acca_legs table ---
    op.create_table(
        "acca_legs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "ticket_id",
            sa.BigInteger(),
            sa.ForeignKey("acca_tickets.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "fixture_id",
            sa.BigInteger(),
            sa.ForeignKey("fixtures.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("market", sa.String(), nullable=False),
        sa.Column("bookmaker", sa.String(), nullable=True),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("note", sa.String(), nullable=True),
        sa.Column("result", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("acca_legs")
    op.drop_table("acca_tickets")
