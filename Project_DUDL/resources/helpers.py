import marimo as mo
from .duck.raw_loader import list_datasets, new_table_form, load_df

def selection_form():
    set_selection = mo.ui.dropdown(list_datasets(), label='Select from existing datasets').form()
    return set_selection

def set_selection(drop_list):
    form = mo.md(
   """
   Choose your algorithm parameters:

   {dropdown}
   {button}
   """
    ).batch(dropdown=mo.ui.dropdown(drop_list), button=mo.ui.button(label="Add New")).form()
    return form

def df_selection(form: mo.ui.form):
    if form.value == 'New':
        return new_table_form()
    elif form.value:
        return load_df(form.value)