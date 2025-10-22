"""add featured_picks with snapshot fields

Revision ID: 2983d902fbc4
Revises: 8823fd295c3d
Create Date: 2025-10-22 13:24:06.202436

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2983d902fbc4'
down_revision: Union[str, Sequence[str], None] = '8823fd295c3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
