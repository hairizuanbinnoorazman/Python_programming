"""empty message

Revision ID: 5938e86bd9d9
Revises: 3d87ecb38036
Create Date: 2017-09-17 19:04:19.718097

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5938e86bd9d9'
down_revision = '3d87ecb38036'
branch_labels = None
depends_on = None


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.add_column('user', sa.Column('lol', sa.String(length=64), nullable=True))
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('user', 'lol')
    ### end Alembic commands ###