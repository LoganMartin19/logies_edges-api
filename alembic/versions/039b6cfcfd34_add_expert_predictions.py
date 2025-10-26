"""add expert_predictions

Revision ID: 039b6cfcfd34
Revises: 02a66b808b18
Create Date: 2025-10-26 20:46:56.047683
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '039b6cfcfd34'
down_revision: Union[str, Sequence[str], None] = '02a66b808b18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create expert_predictions table"""
    op.create_table(
        "expert_predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("fixture_id", sa.BigInteger(), sa.ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("home_prob", sa.Numeric()),
        sa.Column("draw_prob", sa.Numeric()),
        sa.Column("away_prob", sa.Numeric()),
        sa.Column("confidence", sa.String()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.UniqueConstraint("fixture_id", "day", name="uq_expertpred_fixture_day"),
        sa.Index("ix_expertpred_fixture_day", "fixture_id", "day"),
    )


def downgrade() -> None:
    """Drop expert_predictions table"""
    op.drop_table("expert_predictions")
