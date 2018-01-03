"""empty message

Revision ID: 3d87ecb38036
Revises: 
Create Date: 2017-09-17 19:01:02.389302

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '3d87ecb38036'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('friendly_id_slugs')
    op.drop_table('schema_migrations')
    op.drop_table('reviews')
    op.drop_table('ar_internal_metadata')
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ar_internal_metadata',
    sa.Column('key', mysql.VARCHAR(length=255), nullable=False),
    sa.Column('value', mysql.VARCHAR(length=255), nullable=True),
    sa.Column('created_at', mysql.DATETIME(), nullable=False),
    sa.Column('updated_at', mysql.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('key'),
    mysql_default_charset=u'utf8',
    mysql_engine=u'InnoDB'
    )
    op.create_table('reviews',
    sa.Column('id', mysql.VARCHAR(collation=u'utf8mb4_bin', length=255), nullable=False),
    sa.Column('slug', mysql.VARCHAR(collation=u'utf8mb4_bin', length=255), nullable=False),
    sa.Column('url_id', mysql.VARCHAR(collation=u'utf8mb4_bin', length=255), nullable=True),
    sa.Column('metadata', mysql.TEXT(collation=u'utf8mb4_bin'), nullable=True),
    sa.Column('created_at', mysql.DATETIME(), nullable=True),
    sa.Column('updated_at', mysql.DATETIME(), nullable=True),
    mysql_collate=u'utf8mb4_bin',
    mysql_default_charset=u'utf8mb4',
    mysql_engine=u'InnoDB'
    )
    op.create_table('schema_migrations',
    sa.Column('version', mysql.VARCHAR(length=255), nullable=False),
    sa.PrimaryKeyConstraint('version'),
    mysql_default_charset=u'utf8',
    mysql_engine=u'InnoDB'
    )
    op.create_table('friendly_id_slugs',
    sa.Column('id', mysql.INTEGER(display_width=11), nullable=False),
    sa.Column('slug', mysql.VARCHAR(collation=u'utf8mb4_bin', length=255), nullable=False),
    sa.Column('sluggable_id', mysql.INTEGER(display_width=11), autoincrement=False, nullable=False),
    sa.Column('sluggable_type', mysql.VARCHAR(collation=u'utf8mb4_bin', length=50), nullable=True),
    sa.Column('scope', mysql.VARCHAR(collation=u'utf8mb4_bin', length=255), nullable=True),
    sa.Column('created_at', mysql.DATETIME(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    mysql_collate=u'utf8mb4_bin',
    mysql_default_charset=u'utf8mb4',
    mysql_engine=u'InnoDB'
    )
    op.drop_table('user')
    ### end Alembic commands ###
