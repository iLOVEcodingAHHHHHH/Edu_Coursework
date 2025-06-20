"""empty message

Revision ID: cf4ae1de6934
Revises: 803916877a08
Create Date: 2025-06-07 08:48:15.964060

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'cf4ae1de6934'
down_revision = '803916877a08'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('items', schema=None) as batch_op:
        batch_op.add_column(sa.Column('description', sa.String(), nullable=True))

    op.execute("UPDATE invoices SET enable_downloads = False")
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('items', schema=None) as batch_op:
        batch_op.drop_column('description')

    # ### end Alembic commands ###
