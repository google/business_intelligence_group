import causalimpact
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import ipywidgets
import datetime
from tqdm.auto import tqdm
import itertools
import random
from decimal import Decimal, ROUND_HALF_UP
from scipy.spatial.distance import euclidean
import fastdtw

from solutions.causal_impact.preprocess import PreProcess


class CausalImpact(PreProcess):
  """CausalImpact analysis and experimental design on CausalImpact.

  CausalImpact Analysis performs a CausalImpact analysis on the given data and
  outputs the results. The experimental design will be based on N partitions,
  similarity, or share, with 1000 iterations of random sampling, and will output
  the three candidate groups with the closest DTW distance. A combination of
  increments and periods will be used to simulate and return which combination
  will result in a significantly different validation.

  Attributes:
    run_causalImpact: Runs CausalImpact on the given case.
    create_causalimpact_object:
    display_causalimpact_result:
    plot_causalimpact:

  Returns:
    The CausalImpact object.
  """

  colors = [
      '#DB4437',
      '#AB47BC',
      '#4285F4',
      '#00ACC1',
      '#0F9D58',
      '#9E9D24',
      '#F4B400',
      '#FF7043',
  ]
  NUM_OF_ITERATION = 1000
  COMBINATION_TARGET = 10
  TREAT_DURATION = [14, 21, 28]
  TREAT_IMPACT = [1, 1.01, 1.03, 1.05, 1.10, 1.15]
  MAX_STRING_LENGTH = 150

  def __init__(self):
    super().__init__()

  def run_causalImpact(self):
    self.ci_objs = []
    try:
      self.ci_obj = self.create_causalimpact_object(
          self.formatted_data,
          self.date_col_name,
          self.pre_period_start.value,
          self.pre_period_end.value,
          self.post_period_start.value,
          self.post_period_end.value,
          self.num_of_seasons.value,
          self.confidence_interval.value,
      )
      self.ci_objs.append(self.ci_obj)
      self._apply_text_style(
          'success',
          '\nSuccess! CausalImpact has been performed. Check the'
          ' results in the next cell.',
      )

    except Exception as e:
      self._apply_text_style('failure', '\n\nFailure!!')
      print('Error: {}'.format(e))
      print('Please check the following:')
      print('* Date source.')
      print('* Date Column Name.')
      print('* Duration of experiment (pre and post).')
      raise Exception('Please check Failure')

  @staticmethod
  def create_causalimpact_object(
      data,
      date_col,
      pre_start,
      pre_end,
      post_start,
      post_end,
      num_of_seasons,
      confidence_interval):
    if data.index.name != date_col: data.set_index(date_col, inplace=True)

    if num_of_seasons == 1:
      causalimpact_object = causalimpact.fit_causalimpact(
          data=data,
          pre_period=(str(pre_start), str(pre_end)),
          post_period=(str(post_start), str(post_end)),
          alpha= 1 - confidence_interval / 100,
      )
    else:
      causalimpact_object = causalimpact.fit_causalimpact(
          data=data,
          pre_period=(str(pre_start), str(pre_end)),
          post_period=(str(post_start), str(post_end)),
          alpha= 1 - confidence_interval / 100,
          model_options=causalimpact.ModelOptions(
              seasons=[
                  causalimpact.Seasons(num_seasons=num_of_seasons),
              ]
          ),
      )
    return causalimpact_object

  def display_causalimpact_result(self):
    print('Test & Control Time Series')
    line = (
        alt.Chart(self.formatted_data.reset_index())
        .transform_fold(list(self.formatted_data.columns))
        .mark_line()
        .encode(
            alt.X(
                self.date_col_name + ':T',
                title=None,
                axis=alt.Axis(format='%Y %b', tickCount=self.tick_count),
            ),
            y=alt.Y('value:Q', axis=alt.Axis(title='kpi')),
            color=alt.Color(
                'key:N',
                legend=alt.Legend(
                    title=None,
                    orient='none',
                    legendY=-20,
                    direction='horizontal',
                    titleAnchor='start',
                ),
                scale=alt.Scale(
                    domain=list(self.formatted_data.columns),
                    range=CausalImpact.colors,
                ),
            ),
        )
        .properties(height=200, width=600)
    )
    rule = (
        alt.Chart(
          pd.DataFrame({
            'Date': [
                str(self.post_period_start.value),
                str(self.post_period_end.value)
                ],
            'color': ['red', 'orange'],
            })
          )
        .mark_rule(strokeDash=[5, 5])
        .encode(x='Date:T', color=alt.Color('color:N', scale=None))
        )
    display((line+rule).properties(height=200, width=600))
    print('=' * 100)

    self.plot_causalimpact(
        self.ci_objs[0],
        self.pre_period_start.value,
        self.pre_period_end.value,
        self.post_period_start.value,
        self.post_period_end.value,
        self.confidence_interval.value,
        self.date_col_name,
        self.tick_count,
        self.purpose_selection.selected_index
    )

  @staticmethod
  def plot_causalimpact(
      causalimpact_object,
      pre_start,
      pre_end,
      tread_start,
      treat_end,
      confidence_interval,
      date_col_name,
      tick_count,
      purpose_selection
    ):
    causalimpact_df = causalimpact_object.series#.copy()
    mape = mean_absolute_percentage_error(
        causalimpact_df['observed'][str(pre_start) : str(pre_end)],
        causalimpact_df['posterior_mean'][str(pre_start) : str(pre_end)],
    )
    threshold = round(1 - confidence_interval / 100, 2)

    line_1 = (
        alt.Chart(causalimpact_df.reset_index())
        .transform_fold([
            'observed',
            'posterior_mean',
        ])
        .mark_line()
        .encode(
            x=alt.X(
                'yearmonthdate(' + date_col_name + ')',
                axis=alt.Axis(
                    title='',
                    labels=False,
                    ticks=False,
                    format='%Y %b',
                    tickCount=tick_count,
                ),
            ),
            y=alt.Y(
                'value:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(title=''),
            ),
            color=alt.Color(
                'key:N',
                legend=alt.Legend(
                    title=None,
                    orient='none',
                    legendY=-20,
                    direction='horizontal',
                    titleAnchor='start',
                ),
                sort=['posterior_mean', 'observed'],
            ),
            strokeDash=alt.condition(
                alt.datum.key == 'posterior_mean',
                alt.value([5, 5]),
                alt.value([0]),
            ),
        )
    )
    area_1 = (
        alt.Chart(causalimpact_df.reset_index())
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X('yearmonthdate(' + date_col_name + ')'),
            y=alt.Y('posterior_lower:Q', scale=alt.Scale(zero=False)),
            y2=alt.Y2('posterior_upper:Q'),
        )
    )
    line_2 = (
        alt.Chart(causalimpact_df.reset_index())
        .mark_line(strokeDash=[5, 5])
        .encode(
            x=alt.X(
                'yearmonthdate(' + date_col_name + ')',
                axis=alt.Axis(
                    title='',
                    labels=False,
                    ticks=False,
                    format='%Y %b',
                    tickCount=tick_count,
                ),
            ),
            y=alt.Y(
                'point_effects_mean:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(title=''),
            ),
        )
    )
    area_2 = (
        alt.Chart(causalimpact_df.reset_index())
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X('yearmonthdate(' + date_col_name + ')'),
            y=alt.Y('point_effects_lower:Q', scale=alt.Scale(zero=False)),
            y2=alt.Y2('point_effects_upper:Q'),
        )
    )
    line_3 = (
        alt.Chart(causalimpact_df.reset_index())
        .mark_line(strokeDash=[5, 5])
        .encode(
            x=alt.X(
                'yearmonthdate(' + date_col_name + ')',
                axis=alt.Axis(title='', format='%Y %b', tickCount=tick_count),
            ),
            y=alt.Y(
                'cumulative_effects_mean:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(title=''),
            ),
        )
    )
    area_3 = (
        alt.Chart(causalimpact_df.reset_index())
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X(
                'yearmonthdate(' + date_col_name + ')',
                axis=alt.Axis(title='')),
            y=alt.Y('cumulative_effects_lower:Q', scale=alt.Scale(zero=False),
                    axis=alt.Axis(title='')),
            y2=alt.Y2('cumulative_effects_upper:Q'),
        )
    )
    zero_line = (
        alt.Chart(pd.DataFrame({'y': [0]}))
        .mark_rule()
        .encode(y='y', color=alt.value('gray'))
    )
    rules = (
        alt.Chart(
            pd.DataFrame({
                'Date': [str(tread_start), str(treat_end)],
                'color': ['red', 'orange'],
            })
        )
        .mark_rule(strokeDash=[5, 5])
        .encode(x='Date:T', color=alt.Color('color:N', scale=None))
    )
    watermark = alt.Chart(pd.DataFrame([1])).mark_text(
        align='center',
        dx=0,
        dy=0,
        fontSize=48,
        text='mock experiment',
        color='red'
      ).encode(
        opacity=alt.value(0.5)
    )
    if purpose_selection == 1:
      cumulative = line_3 + area_3 + rules + zero_line + watermark
    elif causalimpact_object.summary.p_value.average >= threshold:
      cumulative = area_3 + rules + zero_line
    else:
      cumulative = line_3 + area_3 + rules + zero_line
    plot = alt.vconcat(
        (line_1 + area_1 + rules).properties(height=100, width=600),
        (line_2 + area_2 + rules + zero_line).properties(height=100, width=600),
        (cumulative).properties(height=100, width=600),
    )

    tab_data = ipywidgets.Output()
    tab_report = ipywidgets.Output()
    tab_summary = ipywidgets.Output()
    tab_result = ipywidgets.Tab(children = [tab_summary, tab_report, tab_data])
    tab_result.set_title(0, '>> summary')
    tab_result.set_title(1, '>> report')
    tab_result.set_title(2, '>> data')
    with tab_summary:
      print('Approximate model accuracy >> MAPE:{:.2%}'.format(mape))
      if mape <= 0.05:
          PreProcess._apply_text_style(
              'success',
              'Very Good: The difference between actual and predicted values ​​is slight.')
      elif mape <= 0.10:
          PreProcess._apply_text_style(
              'success',
              'Good: The difference between the actual and predicted values ​​is within the acceptable range.')
      elif mape <= 0.15:
          PreProcess._apply_text_style(
              'failure',
              'Medium: he difference between the actual and predicted values ​​ismoderate, so this is only a reference value.')
      else:
          PreProcess._apply_text_style(
              'failure',
              'Bad: The difference between actual and predicted values ​​is large, so we do not recommend using it.')
      if causalimpact_object.summary.p_value.average <= threshold:
          PreProcess._apply_text_style('success', f'\nP-Value is under {threshold}. There is a statistically significant difference.')
      else:
          PreProcess._apply_text_style('failure', f'\nP-Value is over {threshold}. There is not a statistically significant difference.')

      print(causalimpact.summary(
          causalimpact_object,
          output_format='summary',
          alpha= 1 - confidence_interval / 100))
      display(plot)
    with tab_report:
      print(causalimpact.summary(
          causalimpact_object,
          output_format="report",
          alpha= 1 - confidence_interval / 100))
    with tab_data:
      df = causalimpact_object.series
      df.insert(2, 'diff_percentage', df['point_effects_mean'] / df['observed'])
      display(df)
    display(tab_result)

  def run_experimental_design(self):
    if self.date_selection.selected_index == 0:
      self.start_date_value = min(self.formatted_data.index).date()
      self.end_date_value = max(self.formatted_data.index).date()
    else:
      self.start_date_value = self.start_date.value
      self.end_date_value = self.end_date.value

    if self.design_type.selected_index == 0:
      self.distance_data = self._n_part_split(
          self.formatted_data.query(
              '@self.start_date_value <= index <= @self.end_date_value'
              ),
          self.num_of_split.value,
          CausalImpact.NUM_OF_ITERATION
      )
    elif self.design_type.selected_index == 1:
      self.distance_data = self._find_similar(
          self.formatted_data.query(
              '@self.start_date_value <= index <= @self.end_date_value'
              ),
          self.target_columns.value,
          self.num_of_pick_range.value,
          self.num_of_covariate.value
      )
    elif self.design_type.selected_index == 2:
      self.distance_data = self._from_share(
          self.formatted_data.query(
              '@self.start_date_value <= index <= @self.end_date_value'
              ),
          self.target_share.value,
      )
    elif self.design_type.selected_index == 3:
      self.distance_data = self._given_assignment(
          self.target_columns.value,
          self.control_columns.value,
      )
    else:
      self._apply_text_style('failure', '\n\nFailure!!')
      print('Please check the following:')
      print('* There is something wrong with design type.')
      raise Exception('Please check Failure')

    self._visualize_candidate(
        self.formatted_data,
        self.distance_data,
        self.start_date_value,
        self.end_date_value,
        self.date_col_name,
        self.tick_count
    )
    self._generate_choice()

  @staticmethod
  def _n_part_split(dataframe, num_of_split, NUM_OF_ITERATION):
    """n_part_split

    Args:
      dataframe: xxx.
      num_of_split: xxx.
      NUM_OF_ITERATION: xxx.
    """
    distance_data = pd.DataFrame(columns=['distance'])
    num_of_pick = len(dataframe.columns) // num_of_split

    for l in tqdm(range(NUM_OF_ITERATION)):
      col_list = list(dataframe.columns)
      picked_data = pd.DataFrame()

      # random pick
      picks = []
      for s in range(num_of_split):
        random_pick = random.sample(col_list, num_of_pick)
        picks.append(random_pick)
        col_list = [i for i in col_list if i not in random_pick]
      picks[0].extend(col_list)

      for i in range(len(picks)):
        picked_data = pd.concat([
            picked_data,
            pd.DataFrame(dataframe[picks[i]].sum(axis=1), columns=[i])
            ], axis=1)

      # calculate distance
      distance = CausalImpact._calculate_distance(
          picked_data.reset_index(drop=True)
      )
      distance_data.loc[l, 'distance'] = float(distance)
      for j in range(len(picks)):
        distance_data.at[l, j] = str(sorted(picks[j]))

    distance_data = (
        distance_data.drop_duplicates()
        .sort_values('distance')
        .head(3)
        .reset_index(drop=True)
    )
    return distance_data

  @staticmethod
  def _find_similar(
      dataframe,
      target_columns,
      num_of_pick_range,
      num_of_covariate,
      ):
    distance_data = pd.DataFrame(columns=['distance'])
    target_cols = target_columns.replace(', ', ',').split(',')

    # An error occurs when the number of candidates (max num_of_range times
    # num_of_covariates) is greater than num_of_columns excluding target column.
    if (
        len(dataframe.columns) - len(target_cols)
          >= num_of_pick_range[1] * num_of_covariate):
      pass
    else:
      print('Please check the following:')
      print('* There is something wrong with similarity settings.')
      print('* Total number of columns ー the target = {}'.format(
          len(dataframe.columns) - len(target_cols)))
      print('* But your settings are {}(max pick#) × {}(covariate#)'.format(
          num_of_pick_range[1], num_of_covariate))
      print('* Please set it so that it does not exceed.')
      PreProcess._apply_text_style('failure', '▲▲▲▲▲▲\n\n')
      raise Exception('Please check Failure')

    for l in tqdm(range(CausalImpact.NUM_OF_ITERATION)):
      picked_data = pd.DataFrame()
      remained_list = [
          i for i in list(dataframe.columns) if i not in target_cols
      ]
      picks = []
      for s in range(num_of_covariate):
        pick = random.sample(remained_list, random.randrange(
            num_of_pick_range[0], num_of_pick_range[1] + 1, 1
            )
        )
        picks.append(pick)
        remained_list = [
            ele for ele in remained_list if ele not in pick
        ]
      picks.insert(0, target_cols)
      for i in range(len(picks)):
        picked_data = pd.concat([
            picked_data,
            pd.DataFrame(dataframe[picks[i]].sum(axis=1), columns=[i])
            ], axis=1)

      # calculate distance
      distance = CausalImpact._calculate_distance(
          picked_data.reset_index(drop=True)
      )
      distance_data.loc[l, 'distance'] = float(distance)
      for j in range(len(picks)):
        distance_data.at[l, j] = str(sorted(picks[j]))

    distance_data = (
          distance_data.drop_duplicates()
          .sort_values('distance')
          .head(3)
          .reset_index(drop=True)
    )
    return distance_data

  @staticmethod
  def _from_share(
      dataframe,
      target_share
      ):
    distance_data = pd.DataFrame(columns=['distance'])
    combinations = []

    n = CausalImpact.NUM_OF_ITERATION
    while len(combinations) < CausalImpact.COMBINATION_TARGET:
      n -= 1
      picked_col = np.random.choice(
          dataframe.columns,
          # Shareは50%までなので列数を2分割
          random.randint(1, len(dataframe.columns)//2 + 1),
          replace=False)

      # (todo)@rhirota シェアを除外済みか全体か検討
      if float(Decimal(dataframe[picked_col].sum().sum() / dataframe.sum().sum()
                      ).quantize(Decimal('0.1'), ROUND_HALF_UP)) == target_share:
        combinations.append(sorted(set(picked_col)))
      if n == 1:
        PreProcess._apply_text_style('failure', '\n\nFailure!!')
        print('Please check the following:')
        print('* There is something wrong with design type C.')
        print("* You couldn't find the right combination in the repetitions.")
        print('* Please re-try or re-set target share')
        PreProcess._apply_text_style('failure', '▲▲▲▲▲▲\n\n')
        raise Exception('Please check Failure')

    for comb in tqdm(combinations):
      for l in tqdm(
          range(
              CausalImpact.NUM_OF_ITERATION // CausalImpact.COMBINATION_TARGET),
          leave=False):
        picked_data = pd.DataFrame()
        remained_list = [
            i for i in list(dataframe.columns) if i not in comb
        ]
        picks = []
        picks.append(random.sample(remained_list, random.randrange(
            # (todo)@rhirota 最小Pickを検討
            1, len(remained_list), 1
            )
        ))
        picks.insert(0, comb)

        for i in range(len(picks)):
          picked_data = pd.concat([
              picked_data,
              pd.DataFrame(dataframe[picks[i]].sum(axis=1), columns=[i])
              ], axis=1)

      # calculate distance
      distance = CausalImpact._calculate_distance(
          picked_data.reset_index(drop=True)
      )
      distance_data.loc[l, 'distance'] = float(distance)
      for j in range(len(picks)):
        distance_data.at[l, j] = str(sorted(picks[j]))

    distance_data = (
          distance_data.drop_duplicates()
          .sort_values('distance')
          .head(3)
          .reset_index(drop=True)
    )
    return distance_data

  @staticmethod
  def _given_assignment(target_columns, control_columns):
    distance_data = pd.DataFrame(columns=['distance'])
    distance_data.loc[0, 'distance'] = 0
    distance_data.loc[0, 0] = str(target_columns.replace(', ', ',').split(','))
    distance_data.loc[0, 1] = str(control_columns.replace(', ', ',').split(','))
    return distance_data

  @staticmethod
  def _calculate_distance(dataframe):
    total_distance = 0
    scaled_data = pd.DataFrame()
    for col in dataframe:
      scaled_data[col] = (dataframe[col] - dataframe[col].min()) / (
          dataframe[col].max() - dataframe[col].min()
      )
    scaled_data = scaled_data.diff().reset_index().dropna()
    for v in itertools.combinations(list(scaled_data.columns), 2):
      distance, _ = fastdtw.fastdtw(
          scaled_data.loc[:, ['index', v[0]]],
          scaled_data.loc[:, ['index', v[1]]],
          dist=euclidean,
      )
      total_distance = total_distance + distance
    return total_distance

  @staticmethod
  def _visualize_candidate(
      dataframe,
      distance_data,
      start_date_value,
      end_date_value,
      date_col_name,
      tick_count
      ):
    PreProcess._apply_text_style(
          'failure',
          '\nCheck! Experimental Design Parameters.'
          )
    print('* start_date_value: ' + str(start_date_value))
    print('* end_date_value: ' + str(end_date_value))
    print('* columns:')
    l = []
    for i in range(len(dataframe.columns)):
      l.append(dataframe.columns[i])
      if len(str(l)) >= CausalImpact.MAX_STRING_LENGTH:
        print(str(l).translate(str.maketrans({'[': '', ']': '',  "'": ''})))
        l = []
    print('\n')

    sub_tab=[ipywidgets.Output() for i in distance_data.index.tolist()]
    tab_option = ipywidgets.Tab(sub_tab)
    for i in range (len(distance_data.index.tolist())):
        tab_option.set_title(i,"option_{}".format(i+1))
        with sub_tab[i]:
          candidate_df = pd.DataFrame(index=dataframe.index)
          for col in range(len(distance_data.columns) - 1):
            print(
                'col_' + str(col + 1) + ': '+ distance_data.at[i, col].replace(
                    "'", ""))
            candidate_df[col + 1] = list(
                dataframe.loc[:, eval(distance_data.at[i, col])].sum(axis=1)
            )
            print('\n')
          candidate_df = candidate_df.add_prefix('col_')

          candidate_share = pd.DataFrame(
              candidate_df.loc[str(start_date_value):str(end_date_value), :
                               ].sum(),
              columns=['total'])
          candidate_share['daily_average'] = candidate_share['total'] // (
              end_date_value - start_date_value).days
          candidate_share['share'] = candidate_share['total'] / (dataframe.query(
                '@start_date_value <= index <= @end_date_value'
                ).sum().sum())

          try:
            for i in candidate_df.columns:
              stl = STL(candidate_df[i], robust=True).fit()
              candidate_share.loc[i, 'std'] = np.std(stl.seasonal + stl.resid)
            display(
                candidate_share[['daily_average', 'share', 'std']].style.format(
                    {
                        'daily_average': '{:,.0f}',
                        'share': '{:.1%}',
                        'std': '{:,.0f}',
                        }))
          except Exception as e:
            print(e)
            display(
                candidate_share[['daily_average', 'share']].style.format({
                'daily_average': '{:,.0f}',
                'share': '{:.1%}',
                }))

          chart_line = (
              alt.Chart(candidate_df.reset_index())
              .transform_fold(
                  fold=list(candidate_df.columns), as_=['pivot', 'kpi']
              )
              .mark_line()
              .encode(
                  x=alt.X(
                      date_col_name + ':T',
                      title=None,
                      axis=alt.Axis(
                      grid=False, format='%Y %b', tickCount=tick_count
                      ),
                  ),
                  y=alt.Y('kpi:Q'),
                  color=alt.Color(
                    'pivot:N',
                    legend=alt.Legend(
                      title=None,
                      orient='none',
                      legendY=-20,
                      direction='horizontal',
                      titleAnchor='start'),
                    scale=alt.Scale(
                        domain=list(candidate_df.columns),
                        range=CausalImpact.colors)),
                  )
              .properties(width=600, height=200)
          )

          rules = alt.Chart(
              pd.DataFrame(
                  {
                      'Date': [str(start_date_value), str(end_date_value)],
                      'color': ['red', 'orange']
                      })
              ).mark_rule(strokeDash=[5, 5]).encode(
                  x='Date:T',
                  color=alt.Color('color:N', scale=None))

          df_scaled = candidate_df.copy()
          df_scaled[:] = MinMaxScaler().fit_transform(candidate_df)
          chart_line_scaled = (
              alt.Chart(df_scaled.reset_index())
              .transform_fold(
                  fold=list(candidate_df.columns),
                  as_=['pivot', 'kpi']
              )
              .mark_line()
              .encode(
                  x=alt.X(
                      date_col_name + ':T',
                      title=None,
                      axis=alt.Axis(
                      grid=False, format='%Y %b', tickCount=tick_count
                      ),
                  ),
                  y=alt.Y('kpi:Q'),
                  color=alt.Color(
                    'pivot:N',
                    legend=alt.Legend(
                      title=None,
                      orient='none',
                      legendY=-20,
                      direction='horizontal',
                      titleAnchor='start'),
                    scale=alt.Scale(
                        domain=list(candidate_df.columns),
                        range=CausalImpact.colors)),
                  )
              .properties(width=600, height=80)
          )

          df_diff = pd.DataFrame(
              np.diff(candidate_df, axis=0),
              columns=candidate_df.columns.values,
          )
          scatter = (
              alt.Chart(df_diff.reset_index())
              .mark_circle()
              .encode(
                  alt.X(alt.repeat('column'), type='quantitative'),
                  alt.Y(alt.repeat('row'), type='quantitative'),
              )
              .properties(width=80, height=80)
              .repeat(
                  row=df_diff.columns.values,
                  column=df_diff.columns.values,
              )
          )
          display(
              alt.vconcat(chart_line + rules, chart_line_scaled) | scatter)
    display(tab_option)

  def _generate_choice(self):
    self.your_choice = ipywidgets.Dropdown(
        options=['option_1', 'option_2', 'option_3'],
        description='your choice:',
    )
    self.target_col_to_simulate = ipywidgets.SelectMultiple(
        options=['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'],
        description='target col:',
        value=['col_1',],
    )
    self.covariate_col_to_simulate = ipywidgets.SelectMultiple(
        options=['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'],
        description='covatiate col:',
        value=['col_2',],
        style={'description_width': 'initial'},
    )
    display(
        PreProcess._apply_text_style(
            18,
            '⑷ Please select option, test column & control column(s).'),
        ipywidgets.HBox([
            self.your_choice,
            self.target_col_to_simulate,
            self.covariate_col_to_simulate,
        ]),
    )

  def generate_simulation(self):
    self.test_data = self._extract_data_from_choice(
        self.your_choice.value,
        self.target_col_to_simulate.value,
        self.covariate_col_to_simulate.value,
        self.formatted_data,
        self.distance_data,
    )
    self.simulation_params, self.ci_objs = self._execute_simulation(
        self.test_data,
        self.date_col_name,
        self.start_date_value,
        self.end_date_value,
        self.num_of_seasons.value,
        self.confidence_interval.value,
        CausalImpact.TREAT_DURATION,
        CausalImpact.TREAT_IMPACT,
    )
    self._display_simulation_result(
        self.simulation_params,
        self.ci_objs,
        self.estimate_icpa.value,
    )
    self._plot_simulation_result(
        self.simulation_params,
        self.ci_objs,
        self.date_col_name,
        self.tick_count,
        self.purpose_selection.selected_index,
        self.confidence_interval.value,
    )

  @staticmethod
  def _extract_data_from_choice(
      your_choice,
      target_col_to_simulate,
      covariate_col_to_simulate,
      dataframe,
      distance
      ):
      selection_row = int(your_choice.replace('option_', '')) - 1
      selection_cols = [
          [int(t.replace('col_', '')) - 1 for t in list(target_col_to_simulate)],
          [int(t.replace('col_', '')) - 1 for t in list(covariate_col_to_simulate)
          ]]
      test_data = pd.DataFrame(index = dataframe.index)

      test_column = []
      for i in selection_cols[0]:
        test_column.extend(eval(distance.at[selection_row,i]))
      test_data['test'] = dataframe.loc[
                    :, test_column
                ].sum(axis=1)

      for col in selection_cols[1]:
        test_data['col_'+ str(col+1)] = dataframe.loc[
                :, eval(distance.at[selection_row, col])
            ].sum(axis=1)

      print('* test: {}\\n'.format(str(test_column).replace("'", "")))
      print('* covariate')
      for x,i in zip(test_data.columns[1:],selection_cols[1]):
        print('> {}: {}'.format(
            x,
            str(eval(distance.at[selection_row, i]))).replace("'", "")
            )
      return test_data

  @staticmethod
  def _execute_simulation(
      dataframe,
      date_col_name,
      start_date_value,
      end_date_value,
      num_of_seasons,
      confidence_interval,
      TREAT_DURATION,
      TREAT_IMPACT,
    ):
    ci_objs = []
    simulation_params = []
    adjusted_data = dataframe.copy()

    for duration in tqdm(TREAT_DURATION):
      for impact in tqdm(TREAT_IMPACT, leave=False):
          pre_end_date = end_date_value + datetime.timedelta(days=-duration)
          post_start_date = pre_end_date + datetime.timedelta(days=1)
          adjusted_data.loc[
              np.datetime64(post_start_date) : np.datetime64(end_date_value),
              'test',] = (
                  dataframe.loc[
                  np.datetime64(post_start_date) : np.datetime64(end_date_value
                  ),
                  'test',
              ]
              * impact
          )

          ci_obj = CausalImpact.create_causalimpact_object(
              adjusted_data,
              date_col_name,
              start_date_value,
              pre_end_date,
              post_start_date,
              end_date_value,
              num_of_seasons,
              confidence_interval,
          )
          simulation_params.append([
              start_date_value,
              pre_end_date,
              post_start_date,
              end_date_value,
              impact,
              duration,
          ])
          ci_objs.append(ci_obj)
    return simulation_params, ci_objs

  @staticmethod
  def _display_simulation_result(simulation_params, ci_objs, estimate_icpa):
      simulation_df = pd.DataFrame(
          index=[],
          columns=[
              'mock_lift',
              'Days_simulated',
              'Pre_Period_MAPE',
              'Post_Period_MAPE',
              'Total_effect',
              'Average_effect',
              'Required_budget',
              'p_value',
              'predicted_lift'
          ],
      )
      for i in range(len(ci_objs)):
        impact_df = ci_objs[i].series
        impact_dict = {
            'test_period':'('+str(simulation_params[i][5])+'d) '+str(simulation_params[i][2])+'~'+str(simulation_params[i][3]),
            'mock_lift_rate': simulation_params[i][4] - 1,
            'predicted_lift_rate': ci_objs[i].summary.loc['average', 'rel_effect'],
            'Days_simulated': simulation_params[i][5],
            'Pre_Period_MAPE': [
                mean_absolute_percentage_error(
                    impact_df.loc[:, 'observed'][
                        str(simulation_params[i][0]) : str(
                            simulation_params[i][1]
                        )
                    ],
                    impact_df.loc[:, 'posterior_mean'][
                        str(simulation_params[i][0]) : str(
                            simulation_params[i][1]
                        )
                    ],
                )
            ],
            'Post_Period_MAPE': [
                mean_absolute_percentage_error(
                    impact_df.loc[:, 'observed'][
                        str(simulation_params[i][2]) : str(
                            simulation_params[i][3]
                        )
                    ],
                    impact_df.loc[:, 'posterior_mean'][
                        str(simulation_params[i][2]) : str(
                            simulation_params[i][3]
                        )
                    ],
                )
            ],
            'Total_effect': ci_objs[i].summary.loc['cumulative', 'abs_effect'],
            'Average_effect': ci_objs[i].summary.loc['average', 'abs_effect'],
            'Required_budget': [
                ci_objs[i].summary.loc['cumulative', 'abs_effect'] * estimate_icpa
            ],
            'p_value': ci_objs[i].summary.loc['average', 'p_value'],

        }
        simulation_df = pd.concat(
            [simulation_df, pd.DataFrame.from_dict(impact_dict)],
            ignore_index=True,
        )
      display(PreProcess._apply_text_style(
            18,
            'A/A Test: Check the error without intervention'))
      print('> If p_value < 0.05, please suspect "poor model accuracy"(See Pre_Period_MAPE) or "data drift"(See Time Series Chart).\\n')
      display(
          simulation_df.query('mock_lift_rate == 0')[
              ['test_period','Pre_Period_MAPE','Post_Period_MAPE','p_value']
              ].style.format({
                  'Pre_Period_MAPE': '{:.2%}',
                  'Post_Period_MAPE': '{:.2%}',
                  'p_value': '{:,.2f}',
                  }).hide()
              )
      print('\n')
      display(PreProcess._apply_text_style(
            18,
            'Simulation with increments as a mock experiment'))
      for i in simulation_df.Days_simulated.unique():
        print('\n During the last {} days'.format(i))
        display(
            simulation_df.query('mock_lift_rate != 0 & Days_simulated == @i')[
                [
                    'mock_lift_rate',
                    'predicted_lift_rate',
                    'Pre_Period_MAPE',
                    'Total_effect',
                    'Average_effect',
                    'Required_budget',
                    'p_value',
                    ]
            ].style.format({
                'mock_lift_rate': '{:+.0%}',
                'predicted_lift_rate': '{:+.1%}',
                'Pre_Period_MAPE': '{:.2%}',
                'Total_effect': '{:,.2f}',
                'Average_effect': '{:,.2f}',
                'Required_budget': '{:,.0f}',
                'p_value': '{:,.2f}',
            }).hide()
        )

  @staticmethod
  def _plot_simulation_result(
      simulation_params,
      ci_objs,
      date_col_name,
      tick_count,
      purpose_selection,
      confidence_interval,
      ):

    mock_combinations = []
    for i in range(len(simulation_params)):
      mock_combinations.append(
            [
                '{}d:+{:.0%}'.format(
                    simulation_params[i][5],
                    simulation_params[i][4]-1)
            ])
    simulation_tb=[ipywidgets.Output() for tab in mock_combinations]
    tab_simulation = ipywidgets.Tab(simulation_tb)
    for id,name in enumerate(mock_combinations):
      tab_simulation.set_title(id,name)
      with simulation_tb[id]:
        print(
            'Pre Period:{} ~ {}\\nPost Period:{} ~ {}'.format(
                simulation_params[id][0],
                simulation_params[id][1],
                simulation_params[id][2],
                simulation_params[id][3],
            )
        )
        CausalImpact.plot_causalimpact(
            ci_objs[id],
            simulation_params[id][0],
            simulation_params[id][1],
            simulation_params[id][2],
            simulation_params[id][3],
            confidence_interval,
            date_col_name,
            tick_count,
            purpose_selection
        )
    display(tab_simulation)
