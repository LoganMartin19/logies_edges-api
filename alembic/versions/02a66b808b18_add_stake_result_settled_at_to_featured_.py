"""add stake/result/settled_at to featured_picks

Revision ID: 02a66b808b18
Revises: 802e2b5296ef
Create Date: 2025-10-24 22:37:57.368639

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '02a66b808b18'
down_revision: Union[str, Sequence[str], None] = '802e2b5296ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("featured_picks", sa.Column("stake", sa.Float(), nullable=True))
    op.add_column("featured_picks", sa.Column("result", sa.String(), nullable=True))
    op.add_column("featured_picks", sa.Column("settled_at", sa.DateTime(), nullable=True))

def downgrade() -> None:
    op.drop_column("featured_picks", "settled_at")
    op.drop_column("featured_picks", "result")
    op.drop_column("featured_picks", "stake")
