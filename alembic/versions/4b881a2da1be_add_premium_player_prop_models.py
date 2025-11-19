"""add premium + player_prop_models

Revision ID: 4b881a2da1be
Revises: 23ae18175542
Create Date: 2025-11-19 22:16:45.105225

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4b881a2da1be'
down_revision: Union[str, Sequence[str], None] = '23ae18175542'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
