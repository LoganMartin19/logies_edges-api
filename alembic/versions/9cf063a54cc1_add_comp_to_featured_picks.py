"""add comp to featured_picks

Revision ID: 9cf063a54cc1
Revises: 2983d902fbc4
Create Date: 2025-10-22 13:31:47.273105
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9cf063a54cc1"
down_revision: Union[str, Sequence[str], None] = "2983d902fbc4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ✅ Add the 'comp' column to featured_picks if it doesn't already exist
    op.add_column("featured_picks", sa.Column("comp", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # ✅ Remove the 'comp' column
    op.drop_column("featured_picks", "comp")
