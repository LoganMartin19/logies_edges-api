"""add premium + player_prop_models

Revision ID: 4b881a2da1be
Revises: 23ae18175542
Create Date: 2025-11-19 22:16:45.105225
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4b881a2da1be"
down_revision: Union[str, Sequence[str], None] = "23ae18175542"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- users: premium + stripe fields ------------------------------------
    op.add_column(
        "users",
        sa.Column("stripe_customer_id", sa.String(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "is_premium",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "users",
        sa.Column("premium_activated_at", sa.DateTime(), nullable=True),
    )

    # optional: drop the default going forward (keeps existing rows as false)
    op.alter_column("users", "is_premium", server_default=None)

    # --- tipster_picks: premium-only flag -----------------------------------
    op.add_column(
        "tipster_picks",
        sa.Column(
            "is_premium_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.alter_column("tipster_picks", "is_premium_only", server_default=None)

    # --- player_prop_models table -------------------------------------------
    op.create_table(
        "player_prop_models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "player_odds_id",
            sa.Integer(),
            sa.ForeignKey("player_odds.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        ),
        sa.Column("model_source", sa.String(), nullable=False),
        sa.Column("prob", sa.Float(), nullable=False),
        sa.Column("fair_price", sa.Float(), nullable=False),
        sa.Column("edge", sa.Float(), nullable=False),
        sa.Column(
            "is_premium_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # indexes similar to your model
    op.create_index(
        "ix_player_prop_models_player_odds_id",
        "player_prop_models",
        ["player_odds_id"],
    )
    op.create_index(
        "ix_player_prop_models_model_source",
        "player_prop_models",
        ["model_source"],
    )
    op.create_index(
        "ix_player_prop_models_created_at",
        "player_prop_models",
        ["created_at"],
    )


def downgrade() -> None:
    # reverse of upgrade
    op.drop_index("ix_player_prop_models_created_at", table_name="player_prop_models")
    op.drop_index("ix_player_prop_models_model_source", table_name="player_prop_models")
    op.drop_index("ix_player_prop_models_player_odds_id", table_name="player_prop_models")
    op.drop_table("player_prop_models")

    op.drop_column("tipster_picks", "is_premium_only")

    op.drop_column("users", "premium_activated_at")
    op.drop_column("users", "is_premium")
    op.drop_column("users", "stripe_customer_id")
