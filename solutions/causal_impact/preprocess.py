# Data Load
from google.colab import auth, files, widgets
from google.auth import default
from google.cloud import bigquery
import io
import gspread
from oauth2client.client import GoogleCredentials

# Calculate
import altair as alt
import itertools
import random
import numpy as np
import pandas as pd
import fastdtw

from tslearn.clustering import TimeSeriesKMeans
from decimal import Decimal, ROUND_HALF_UP
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

# UI/UX
import datetime
from dateutil.relativedelta import relativedelta
import ipywidgets
from IPython.display import display, Markdown, HTML, Javascript
from tqdm.auto import tqdm
import warnings
warnings.simplefilter('ignore')

class PreProcess(object):
  """PreProcess handles process from data loading to visualization.

    Create a UI, load time series data based on input and do some
    transformations to pass it to analysis. This also includes visualization of
    points that should be confirmed in time series data.

    Attributes:
      _apply_text_style: Decorate the text
      define_ui: Define the UI using ipywidget
      generate_ui: Generates UI for input from the user
      load_data: Load data from any data source
      _load_data_from_sheet: Load data from spreadsheet
      _load_data_from_csv: Load data from CSV
      _load_data_from_bigquery: Load data from Big Query
      format_date: Set index
      _shape_wide: Configure narrow/wide conversion
      _trend_check: Visualize data
      saving_params: Save the contents entered in the UI
      set_params: Set the saved input contents to the instance
  """

  def __init__(self):
    self.define_ui()

  @staticmethod
  def _apply_text_style(type, text):
    # todo@(rhirota): Need to reconsideration about type
    if type == 'success':
      return print(f"\033[38;2;15;157;88m " + text + "\033[0m")

    if type == 'failure':
      return print(f"\033[38;2;219;68;55m " + text + "\033[0m")

    if isinstance(type, int):
      span_style = ipywidgets.HTML(
        "<span style='font-size:" + str(type) + "px; background: "
        "linear-gradient(transparent 90%, #4285F4 0%);'>"
        + text
        + '</style>'
      )
      return span_style

  def define_ui(self):
    # Input box for data sources
    self.sheet_url = ipywidgets.Text(
        placeholder='Please enter google spreadsheet url',
        value='https://docs.google.com/spreadsheets/d/1dISrbX1mZHgzpsIct2QXFOWWRRJiCxDSmSzjuZz64Tw/edit#gid=0',
        description='spreadsheet url:',
        style={'description_width': 'initial'},
        layout=ipywidgets.Layout(width='800px'),
    )
    self.sheet_name = ipywidgets.Text(
        placeholder='Please enter sheet name',
        value='analysis_data',
        # value='raw_data',
        description='sheet name:',
    )
    self.csv_name = ipywidgets.Text(
        placeholder='Please enter csv name',
        description='csv name:',
        layout=ipywidgets.Layout(width='500px'),
    )
    self.bq_project_id = ipywidgets.Text(
        placeholder='Please enter project id',
        description='project id:',
        layout=ipywidgets.Layout(width='500px'),
    )
    self.bq_table_name = ipywidgets.Text(
        placeholder='Please enter table name',
        description='table name:',
        layout=ipywidgets.Layout(width='500px'),
    )

    # Input box for data format
    self.date_col = ipywidgets.Text(
        placeholder='Please enter date column name',
        value='Date',
        description='date column:',
    )
    self.pivot_col = ipywidgets.Text(
        placeholder='Please enter pivot column name',
        value='Geo',
        description='pivot column:',
    )
    self.kpi_col = ipywidgets.Text(
        placeholder='Please enter kpi column name',
        value='KPI',
        description='kpi column:',
    )

    # Input box for Date-related
    self.pre_period_start = ipywidgets.DatePicker(
        description='Pre Start:',
        value=datetime.date.today() - relativedelta(days=122),
    )
    self.pre_period_end = ipywidgets.DatePicker(
        description='Pre End:',
        value=datetime.date.today() - relativedelta(days=32),
    )
    self.post_period_start = ipywidgets.DatePicker(
        description='Post Start:',
        value=datetime.date.today() - relativedelta(days=31),
    )
    self.post_period_end = ipywidgets.DatePicker(
        description='Post End:',
        value=datetime.date.today(),
    )
    self.start_date = ipywidgets.DatePicker(
        description='Start Date:',
        value=datetime.date.today() - relativedelta(days=122),
    )
    self.end_date = ipywidgets.DatePicker(
        description='End Date:',
        value=datetime.date.today() - relativedelta(days=32),
    )
    self.depend_data = ipywidgets.ToggleButton(
        value=False,
        description='Click >> Use the beginning and end of data',
        disabled=False,
        button_style='info',
        tooltip='Description',
        layout=ipywidgets.Layout(width='300px'),
    )

    # Input box for Experimental_Design-related
    self.exclude_cols = ipywidgets.Text(
        placeholder=(
            'Enter comma-separated columns if any columns are not used in the'
            ' design.'
        ),
        description='exclude cols:',
        layout=ipywidgets.Layout(width='1000px'),
    )
    self.num_of_split = ipywidgets.Dropdown(
        options=[2, 3, 4, 5],
        value=2,
        description='split#:',
        disabled=False,
    )
    self.target_columns = ipywidgets.Text(
        placeholder='Please enter comma-separated entries',
        value='Tokyo, Kanagawa',
        description='target_cols:',
        layout=ipywidgets.Layout(width='500px'),
    )
    self.num_of_pick_range = ipywidgets.IntRangeSlider(
        value=[5, 10],
        min=1,
        max=50,
        step=1,
        description='pick range:',
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    self.num_of_covariate = ipywidgets.Dropdown(
        options=[1, 2, 3, 4, 5],
        value=1,
        description='covariate#:',
        layout=ipywidgets.Layout(width='192px'),
    )
    self.target_share = ipywidgets.FloatSlider(
        value=0.3,
        min=0.05,
        max=0.5,
        step=0.05,
        description='target share#:',
        orientation='horizontal',
        readout=True,
        readout_format='.1%',
    )
    self.control_columns = ipywidgets.Text(
        placeholder='Please enter comma-separated entries',
        value='Aomori, Akita',
        description='control_cols:',
        layout=ipywidgets.Layout(width='500px'),
    )

    # Input box for simulation params
    self.num_of_seasons = ipywidgets.IntText(
        value=1,
        description='num_of_seasons:',
        disabled=False,
        style={'description_width': 'initial'},
    )
    self.estimate_icpa = ipywidgets.IntText(
        value=1000,
        description='Estimated iCPA:',
        style={'description_width': 'initial'},
    )
    self.confidence_interval = ipywidgets.RadioButtons(
        options=[80, 90, 95],
        value=90,
        description='Confidence interval %:',
        style={'description_width': 'initial'},
    )

  def generate_ui(self):
    # UI for data soure
    self.soure_selection = ipywidgets.Tab()
    self.soure_selection.children = [
        ipywidgets.VBox([self.sheet_url, self.sheet_name]),
        ipywidgets.VBox([self.csv_name]),
        ipywidgets.VBox([self.bq_project_id, self.bq_table_name]),
    ]
    self.soure_selection.set_title(0, 'Google_Spreadsheet')
    self.soure_selection.set_title(1, 'CSV_file')
    self.soure_selection.set_title(2, 'Big_Query')

    # UI for data type(narrow or wide)
    self.data_type_selection = ipywidgets.Tab()
    self.data_type_selection.children = [
        ipywidgets.VBox([
            ipywidgets.Label(
                'Wide, or unstacked data is presented with each different'
                ' data variable in a separate column.'
            ),
            self.date_col,
        ]),
        ipywidgets.VBox([
            ipywidgets.Label(
                'Narrow, stacked, or long data is presented with one column '
                'containing all the values and another column listing the '
                'context of the value'
            ),
            ipywidgets.HBox([self.date_col, self.pivot_col, self.kpi_col]),
        ]),
    ]
    self.data_type_selection.set_title(0, 'Wide_Format')
    self.data_type_selection.set_title(1, 'Narrow_Format')

    # UI for experimental design
    self.design_type = ipywidgets.Tab(
        children=[
            ipywidgets.VBox([
                ipywidgets.HTML(
                    'divide_equally divides the time series data into N'
                    ' groups(split#) with similar movements.'
                ),
                self.num_of_split,
                self.exclude_cols,
            ]),
            ipywidgets.VBox([
                ipywidgets.HTML(
                    'similarity_selection extracts N groups(covariate#) that '
                    'move similarly to particular columns(target_cols).'
                ),
                ipywidgets.HBox([
                    self.target_columns,
                    self.num_of_covariate,
                    self.num_of_pick_range,
                ]),
                self.exclude_cols,
            ]),
            ipywidgets.VBox([
                ipywidgets.HTML(
                    'target share extracts targeted time series data from'
                    ' the proportion of interventions.'
                ),
                self.target_share,
                self.exclude_cols,
            ]),
            ipywidgets.VBox([
                ipywidgets.HTML(
                    'To improve reproducibility, it is important to create an'
                    ' accurate counterfactual model rather than a balanced'
                    ' assignment.'
                ),
                self.target_columns,
                self.control_columns,
            ]),
        ]
    )
    self.design_type.set_title(0, 'A: divide_equally')
    self.design_type.set_title(1, 'B: similarity_selection')
    self.design_type.set_title(2, 'C: target_share')
    self.design_type.set_title(3, 'D: pre-allocated')

    # UI for purpose (CausalImpact or Experimental Design)
    self.purpose_selection = ipywidgets.Tab()
    self.date_selection = ipywidgets.Tab()
    self.date_selection.children = [
        ipywidgets.VBox(
            [
                ipywidgets.HTML('The <b>minimum</b> date of the data is '
                'selected as the start date.'),
                ipywidgets.HTML('The <b>maximum</b> date in the data is '
                'selected as the end date.'),
            ]),
        ipywidgets.VBox(
            [
                self.start_date,
                self.end_date,
            ]
        )]
    self.date_selection.set_title(0, 'automatic selection')
    self.date_selection.set_title(1, 'manual input')

    self.purpose_selection.children = [
        # Causalimpact
        ipywidgets.VBox([
            PreProcess._apply_text_style(
                15, '⑶ - a: Enter the Pre and Post the intervention.'
            ),
            self.pre_period_start,
            self.pre_period_end,
            self.post_period_start,
            self.post_period_end,
            PreProcess._apply_text_style(
                15,
                '⑶ - b: Enter the number of periodicities in the'
                ' time series data.(default=1)',
            ),
            ipywidgets.VBox([self.num_of_seasons, self.confidence_interval]),
        ]),
        # Experimental_Design
        ipywidgets.VBox([
            PreProcess._apply_text_style(
                15,
                '⑶ - a: Please select date for experimental design',
            ),
            self.date_selection,
            PreProcess._apply_text_style(
                15,
                '⑶ - b: Select the <b>experimental design method</b> and'
                ' enter the necessary items.',
            ),
            self.design_type,
            PreProcess._apply_text_style(
                15,
                '⑶ - c: (Optional) Enter <b>Estimated incremental CPA</b>(Cost'
                ' of intervention ÷ Lift from intervention without bias) & the '
                'number of periodicities in the time series data.',
            ),
            ipywidgets.VBox([
                self.estimate_icpa,
                self.num_of_seasons,
                self.confidence_interval,
            ]),
        ]),
    ]
    self.purpose_selection.set_title(0, 'Causalimpact')
    self.purpose_selection.set_title(1, 'Experimental_Design')

    display(
        PreProcess._apply_text_style(18, '⑴ Please select a data source.'),
        self.soure_selection,
        Markdown('<br>'),
        PreProcess._apply_text_style(
            18, '⑵ Please select wide or narrow data format.'
        ),
        self.data_type_selection,
        Markdown('<br>'),
        PreProcess._apply_text_style(
            18, '⑶ Please select the purpose and set conditions.'
        ),
        self.purpose_selection,
    )

  def load_data(self):
    if self.soure_selection.selected_index == 0:
      try:
        self.loaded_df = self._load_data_from_sheet(
            self.sheet_url.value, self.sheet_name.value
        )
      except Exception as e:
        self._apply_text_style('failure', '\n\nFailure!!')
        print('Error: {}'.format(e))
        print('Please check the following:')
        print('* sheet url:{}'.format(self.sheet_url.value))
        print('* sheet name:{}'.format(self.sheet_name.value))
        raise Exception('Please check Failure')

    elif self.soure_selection.selected_index == 1:
      try:
        self.loaded_df = self._load_data_from_csv(self.csv_name.value)
      except Exception as e:
        self._apply_text_style('failure', '\n\nFailure!!')
        print('Error: {}'.format(e))
        print('Please check the following:')
        print('* There is something wrong with the CSV-related settings.')
        print('* CSV namel:{}'.format(self.csv_name.value))
        raise Exception('Please check Failure')

    elif self.soure_selection.selected_index == 2:
      try:
        self.loaded_df = self._load_data_from_bigquery(
            self.bq_project_id.value, self.bq_table_name.value
        )
      except Exception as e:
        self._apply_text_style('failure', '\n\nFailure!!')
        print('Error: {}'.format(e))
        print('Please check the following:')
        print('* There is something wrong with the bq-related settings.')
        print('* bq project id:{}'.format(self.bq_project_id.value))
        print('* bq table name :{}'.format(self.bq_table_name.value))
        raise Exception('Please check Failure')

    else:
      raise Exception('Please select a data souce at Step.1-2.')

    self._apply_text_style(
        'success',
        'Success! The target data has been loaded.')
    display(self.loaded_df.head(3))

  @staticmethod
  def _load_data_from_sheet(spreadsheet_url, sheet_name):
    """load_data_from_sheet load data from spreadsheet.

    Args:
      spreadsheet_url: Spreadsheet url with data.
      sheet_name: Sheet name with data.
    """
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)
    _workbook = gc.open_by_url(spreadsheet_url)
    _worksheet = _workbook.worksheet(sheet_name)
    df_sheet = pd.DataFrame(_worksheet.get_all_values())
    df_sheet.columns = list(df_sheet.loc[0, :])
    df_sheet.drop(0, inplace=True)
    df_sheet.reset_index(drop=True, inplace=True)
    df_sheet.replace(',', '', regex=True, inplace=True)
    df_sheet.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    df_sheet = df_sheet.apply(pd.to_numeric, errors='ignore')
    return df_sheet

  @staticmethod
  def _load_data_from_csv(csv_name):
    """load_data_from_csv read data from csv.

    Args:
    csv_name: csv file name.
    """
    uploaded = files.upload()
    df_csv = pd.read_csv(io.BytesIO(uploaded[csv_name]))
    df_csv.replace(',', '', regex=True, inplace=True)
    df_csv.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    df_csv = df_csv.apply(pd.to_numeric, errors='ignore')
    return df_csv

  @staticmethod
  def _load_data_from_bigquery(bq_project_id, bq_table_name):
    """_load_data_from_bigquery load data from bigquery.

    Args:
    bq_project_id: bigquery project id.
    bq_table_name: bigquery table name
    """
    auth.authenticate_user()
    client = bigquery.Client(project=bq_project_id)
    query = 'SELECT * FROM `' + bq_table_name + '`;'
    df_bq = client.query(query).to_dataframe()
    df_bq.replace(',', '', regex=True, inplace=True)
    df_bq.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    df_bq = df_bq.apply(pd.to_numeric, errors='ignore')
    return df_bq

  def format_data(self):
    # Remove spaces from input data
    self.date_col_name = self.date_col.value.replace(' ', '')
    self.pivot_col_name = self.pivot_col.value.replace(' ', '')
    self.kpi_col_name = self.kpi_col.value.replace(' ', '')

    try:
      if self.data_type_selection.selected_index == 0:
        self.formatted_data = self.loaded_df.copy()
      elif self.data_type_selection.selected_index == 1:
        self.formatted_data = self._shape_wide(
            self.loaded_df,
            self.date_col_name,
            self.pivot_col_name,
            self.kpi_col_name,
        )

      self.formatted_data.drop(
          self.exclude_cols.value.replace(', ', ',').split(','),
          axis=1,
          errors='ignore',
          inplace=True,
      )
      self.formatted_data[self.date_col_name] = pd.to_datetime(
          self.formatted_data[self.date_col_name]
      )
      self.formatted_data = self.formatted_data.set_index(self.date_col_name)
      self.formatted_data = self.formatted_data.reindex(
          pd.date_range(
              start=self.formatted_data.index.min(),
              end=self.formatted_data.index.max(),
              name=self.formatted_data.index.name))
      self.tick_count = len(self.formatted_data.resample('M')) - 1
      self._apply_text_style(
          'success',
          '\nSuccess! The data was formatted for analysis.'
          )
      display(self.formatted_data.head(3))
      self._apply_text_style(
          'failure',
          '\nCheck! Here is an overview of the data.'
          )
      print(
          'Index name:{} | The earliest date: {} | The latest date: {}'.format(
              self.formatted_data.index.name,
              min(self.formatted_data.index),
              max(self.formatted_data.index)
              ))
      print('* Rows with missing values')
      self.missing_row = self.formatted_data[
          self.formatted_data.isnull().any(axis=1)]
      if len(self.missing_row) > 0:
        self.missing_row
      else:
        print('>> Does not include missing values')

      self._apply_text_style(
          'failure',
          '\nCheck! below [total_trend] / [each_trend] / [describe_data]'
          )
      self._trend_check(
          self.formatted_data,
          self.date_col_name,
          self.tick_count)

    except Exception as e:
      self._apply_text_style('failure', '\n\nFailure!!')
      print('Error: {}'.format(e))
      self._apply_text_style('failure', '\nPlease check the following:')
      if self.data_type_selection.selected_index == 0:
        print('* Your selected data format: Wide format at (2)')
        print('1. Check if the data source is wide.')
        print('2. Compare \"date column\"( {} ) and \"data source\"'.format(
            self.date_col.value))
        print('\n\n')
      else:
        print('* Your selected data format: Narrow format at (2)')
        print('1. Check if the data source is narrow.')
        print('2. Compare \"your input\" and \"data source')
        print('>> date column: {}'.format(self.date_col.value))
        print('>> pivot column: {}'.format(self.pivot_col.value))
        print('>> kpi column: {}'.format(self.kpi_col.value))
        print('\n\n')
      raise Exception('Please check Failure')

  @staticmethod
  def _shape_wide(dataframe, date_column, pivot_column, kpi_column):
    """shape_wide pivots the data in the specified column.

    Converts long data to wide data suitable for experiment design.

    Args:
        dataframe: The DataFrame to be pivoted.
        date_column: The name of the column that contains the dates.
        pivot_column: The name of the column that contains the pivot keys.
        kpi_column: The name of the column that contains the KPI values.

    Returns:
        A DataFrame with the pivoted data.
    """
    # Check if the pivot_column is a single column or a list of columns.
    if ',' in pivot_column:
      group_cols = pivot_column.replace(', ', ',').split(',')
    else:
      group_cols = [pivot_column]

    pivoted_df = pd.pivot_table(
        (dataframe[[date_column] + [kpi_column] + group_cols])
        .groupby([date_column] + group_cols)
        .sum(),
        index=date_column,
        columns=group_cols,
        fill_value=0,
    )
    # Drop the first level of the column names.
    pivoted_df.columns = pivoted_df.columns.droplevel(0)
    # If there are multiple columns, convert the column names to a single string.
    if len(pivoted_df.columns.names) > 1:
      new_cols = [
          '_'.join([x.replace(',', '_') for x in y])
          for y in pivoted_df.columns.values
      ]
      pivoted_df.columns = new_cols
    pivoted_df = pivoted_df.reset_index()
    return pivoted_df

  @staticmethod
  def _trend_check(dataframe, date_col_name, tick_count):
    """trend_check visualize daily trend, 7-day moving average

    Args:
      dataframe: Wide data to check the trend
      date_col_name: xxx
    """
    df_each = pd.DataFrame(index=dataframe.index)
    col_list = list(dataframe.columns)
    for i in col_list:
      min_max = (
          dataframe[i] - dataframe[i].min()
          ) / (dataframe[i].max() - dataframe[i].min())
      df_each = pd.concat([df_each, min_max], axis = 1)

    metric = 'dtw'
    n_clusters = 5
    tskm_base = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,
                             max_iter=100, random_state=42)
    df_cluster = pd.DataFrame({
        "pivot": col_list,
        "cluster": tskm_base.fit_predict(df_each.T).tolist()})
    cluster_counts = (
        df_cluster["cluster"].value_counts().sort_values(ascending=True))

    cluster_text = []
    line_each = []
    for i in cluster_counts.index:
      clust_list = df_cluster.query("cluster == @i")["pivot"].to_list()
      source = df_each.filter(items=clust_list)
      cluster_text.append(str(clust_list).translate(
          str.maketrans({'[': '', ']': '',  "'": ''})))
      line_each.append(
          alt.Chart(source.reset_index())
          .transform_fold(fold=clust_list, as_=['pivot', 'kpi'])
          .mark_line()
          .encode(
              alt.X(
                  date_col_name + ':T',
                  title=None,
                  axis=alt.Axis(
                      grid=False, format='%Y %b', tickCount=tick_count
                      ),
                  ),
              alt.Y('kpi:Q', stack=None, axis=None),
              alt.Color(str(i) + ':N', title=None, legend=None),
              alt.Row(
                  'pivot:N',
                  title=None,
                  header=alt.Header(labelAngle=0, labelAlign='left'),
                  ),
              )
          .properties(bounds='flush', height=30)
          .configure_facet(spacing=0)
          .configure_view(stroke=None)
          .configure_title(anchor='end')
          )

    df_long = (
        pd.melt(dataframe.reset_index(), id_vars=date_col_name)
        .groupby(date_col_name)
        .sum(numeric_only=True)
        .reset_index()
    )
    line_total = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=alt.X(
                date_col_name + ':T',
                axis=alt.Axis(
                    title='', format='%Y %b', tickCount=tick_count
                ),
            ),
            y=alt.Y('value:Q', axis=alt.Axis(title='kpi')),
            color=alt.value('#4285F4'),
        )
    )
    moving_average = (
        alt.Chart(df_long)
        .transform_window(
            rolling_mean='mean(value)',
            frame=[-4, 3],
        )
        .mark_line()
        .encode(
            x=alt.X(date_col_name + ':T'),
            y=alt.Y('rolling_mean:Q'),
            color=alt.value('#DB4437'),
        )
    )
    tab_total_trend = ipywidgets.Output()
    tab_each_trend = ipywidgets.Output()
    tab_describe_data = ipywidgets.Output()
    tab_result = ipywidgets.Tab(children = [
        tab_total_trend,
        tab_each_trend,
        tab_describe_data,
        ])
    tab_result.set_title(0, '>> total_trend')
    tab_result.set_title(1, '>> each_trend')
    tab_result.set_title(2, '>> describe_data')
    display(tab_result)
    with tab_total_trend:
      display(
          (line_total + moving_average).properties(
              width=700,
              height=200,
              title={
                  'text': ['Daily Trend(blue) & 7days moving average(red)'],
              },
          )
      )
    with tab_each_trend:
      for i in range(len(cluster_text)):
          print('cluster {}:{}'.format(i, cluster_text[i]))
          display(line_each[i].properties(width=700))
    with tab_describe_data:
      display(dataframe.describe(include='all'))

  @staticmethod
  def saving_params(instance):
    params_dict = {
        # section for data source
        'soure_selection': instance.soure_selection.selected_index,
        'sheet_url': instance.sheet_url.value,
        'sheet_name': instance.sheet_name.value,
        'csv_name': instance.csv_name.value,
        'bq_project_id': instance.bq_project_id.value,
        'bq_table_name': instance.bq_table_name.value,

        # section for data format(narrow or wide)
        'data_type_selection': instance.data_type_selection.selected_index,
        'date_col': instance.date_col.value,
        'pivot_col': instance.pivot_col.value,
        'kpi_col': instance.kpi_col.value,

        # section for porpose(CausalImpact or Experimental Design)
        'purpose_selection': instance.purpose_selection.selected_index,
        'pre_period_start': instance.pre_period_start.value,
        'pre_period_end': instance.pre_period_end.value,
        'post_period_start': instance.post_period_start.value,
        'post_period_end': instance.post_period_end.value,
        'start_date': instance.start_date.value,
        'end_date': instance.end_date.value,
        'depend_data': instance.depend_data.value,

        'design_type': instance.design_type.selected_index,
        'num_of_split': instance.num_of_split.value,
        'target_columns': instance.target_columns.value,
        'control_columns': instance.control_columns.value,
        'num_of_pick_range': instance.num_of_pick_range.value,
        'num_of_covariate': instance.num_of_covariate.value,
        'target_share': instance.target_share.value,
        'exclude_cols': instance.exclude_cols.value,

        'num_of_seasons': instance.num_of_seasons.value,
        'estimate_icpa': instance.estimate_icpa.value,
        'confidence_interval': instance.confidence_interval.value,
        }
    return params_dict

  @staticmethod
  def set_params(instance, dict_params):
    # section for data source
    instance.soure_selection.selected_index = dict_params['soure_selection']
    instance.sheet_url.value = dict_params['sheet_url']
    instance.sheet_name.value = dict_params['sheet_name']
    instance.csv_name.value = dict_params['csv_name']
    instance.bq_project_id.value = dict_params['bq_project_id']
    instance.bq_table_name.value = dict_params['bq_table_name']

    # section for data format(narrow or wide)
    instance.data_type_selection.selected_index = dict_params['data_type_selection']
    instance.date_col.value = dict_params['date_col']
    instance.pivot_col.value = dict_params['pivot_col']
    instance.kpi_col.value = dict_params['kpi_col']

    # section for porpose(CausalImpact or Experimental Design)
    instance.purpose_selection.selected_index = dict_params['purpose_selection']
    instance.pre_period_start.value = dict_params['pre_period_start']
    instance.pre_period_end.value = dict_params['pre_period_end']
    instance.post_period_start.value = dict_params['post_period_start']
    instance.post_period_end.value = dict_params['post_period_end']
    instance.start_date.value = dict_params['start_date']
    instance.end_date.value = dict_params['end_date']
    instance.depend_data.value = dict_params['depend_data']

    instance.design_type.selected_index = dict_params['design_type']
    instance.num_of_split.value = dict_params['num_of_split']
    instance.target_columns.value = dict_params['target_columns']
    instance.control_columns.value = dict_params['control_columns']
    instance.num_of_pick_range.value = dict_params['num_of_pick_range']
    instance.num_of_covariate.value = dict_params['num_of_covariate']
    instance.target_share.value = dict_params['target_share']
    instance.exclude_cols.value = dict_params['exclude_cols']

    instance.num_of_seasons.value = dict_params['num_of_seasons']
    instance.estimate_icpa.value = dict_params['estimate_icpa']
    instance.confidence_interval.value = dict_params['confidence_interval']
