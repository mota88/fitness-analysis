import pandas
import numpy
import matplotlib.pyplot
import seaborn
import plotly.express
import plotly.graph_objects
import plotly.subplots
from datetime import timedelta


def read_data(file_path):
    """
    Reads the CSV file into a pandas DataFrame and parses the 'Fecha' column as datetime.
    """
    data = pandas.read_csv(file_path, parse_dates=['Fecha'])
    return data


def filter_data_by_date(data, start_date, end_date):
    """
    Filters the DataFrame to only include rows from the start_date to the end_date.
    """
    filtered_data = data[(data['Fecha'] >= start_date) & (data['Fecha'] <=end_date)].reset_index(drop=True)
    return filtered_data


def split_and_aggregate_phases(data, maintenance_duration, pre_deficit_duration, phases_duration):
    """
    Splits the data into phases, aggregates the data by the mean for each phase, and returns a single DataFrame
    with each phase as a row. Adds a 'Phase' column to indicate the phase name and modifies the 'Fecha' column
    to show the date range for each phase. The 'Phase' column is placed as the first column.

    The last phase may have a variable duration if there are fewer days remaining than the specified phase duration.
    """
    total_days = len(data)
    phase_durations = [maintenance_duration, pre_deficit_duration]
    remaining_days = total_days - maintenance_duration - pre_deficit_duration

    # calculate the number of full phases and any remaining days for the last phase
    full_phases = remaining_days // phases_duration
    last_phase_days = remaining_days % phases_duration

    # build phase durations list
    phase_durations += [phases_duration] * full_phases
    if last_phase_days > 0:
        phase_durations.append(last_phase_days)

    phases_list = []
    start_index = 0

    # generate phase names
    phase_names = ['Mantenimiento', 'Pre-déficit'] + [f'Fase déficit {i + 1}' for i in range(len(phase_durations) - 2)]

    for phase_name, duration in zip(phase_names, phase_durations):
        end_index = start_index + duration
        phase_data = data.iloc[start_index:end_index].copy()

        # aggregate the data by taking the mean for each column
        aggregated_data = phase_data.mean()
        aggregated_data['Fase'] = phase_name

        # format the date range for the 'Fecha' column
        start_date = phase_data['Fecha'].iloc[0].strftime('%d-%m-%Y')
        end_date = phase_data['Fecha'].iloc[-1].strftime('%d-%m-%Y')
        aggregated_data['Fecha'] = f"{start_date} - {end_date}"

        # append the aggregated data as a row
        phases_list.append(aggregated_data)

        start_index = end_index

    # convert the list of Series into a DataFrame
    phases_df = pandas.DataFrame(phases_list).reset_index(drop=True)

    # reorder columns to place 'Phase' as the first column
    columns_order = ['Fase'] + [col for col in phases_df.columns if col != 'Fase']
    phases_df = phases_df[columns_order]

    return phases_df


def compute_weight_evolution(df):
    """
    Computes the evolution of weight for data given in a DataFrame.
    """
    # calculate the percentage weight change for this interval
    initial_weight = df['Peso'].iloc[0]
    final_weight = df['Peso'].iloc[-1]
    weight_evolution = ((final_weight - initial_weight) / initial_weight) * 100

    return round(weight_evolution, 2)


def calculate_weight_percentage_change_per_phase(filtered_data, phases_df):
    """
    Calculates the percentage change in weight within each phase using the daily data from filtered_data.
    Adds a new column 'Weight Change (%)' to the phases_df DataFrame, placed after the 'Peso' column.
    """
    weight_changes = []

    for _, phase_row in phases_df.iterrows():
        # extract the date range for the current phase
        start_date, end_date = phase_row['Fecha'].split(' - ')
        list_of_numbers_start = start_date.split('-')
        list_of_numbers_end = end_date.split('-')
        start_date = pandas.to_datetime(str(list_of_numbers_start[2]) + '-' + str(list_of_numbers_start[1]) + '-' + str(list_of_numbers_start[0]))
        end_date = pandas.to_datetime(str(list_of_numbers_end[2]) + '-' + str(list_of_numbers_end[1]) + '-' + str(list_of_numbers_end[0]))
        phase_data = filtered_data[(filtered_data['Fecha'] >= start_date) & (filtered_data['Fecha'] <= end_date)]

        # calculate percentage change in weight for the phase
        weight_evolution = compute_weight_evolution(phase_data)

        weight_changes.append(weight_evolution)

    # insert the weight changes as a new column after the 'Peso' column
    insert_position = phases_df.columns.get_loc('Peso') + 1
    phases_df.insert(insert_position, 'Evolución de Peso (%)', weight_changes)

    return phases_df


def calculate_total_evolution(filtered_data, phases_df, deficit_start_date):
    """
    Calculates the total evolution of all relevant fields from the start of Deficit 1 phase
    to the last recorded day in filtered_data.
    Adds a new row to the phases_df DataFrame with the calculated values.
    """
    # filter the data from the start of Deficit 1 phase to the last recorded day
    total_data = filtered_data[filtered_data['Fecha'] >= deficit_start_date]

    # calculate the means of relevant fields in the selected interval
    total_means = total_data.mean(numeric_only=True)

    # format the date range for the new row
    start_date = total_data['Fecha'].iloc[0].strftime('%d-%m-%Y')
    end_date = total_data['Fecha'].iloc[-1].strftime('%d-%m-%Y')
    date_range = f"{start_date} - {end_date}"

    # create a new row with the calculated values
    total_row = {
        'Fase': 'Déficit completo',
        'Fecha': date_range
    }

    # add the calculated values to the new row
    total_row.update(total_means.to_dict())

    # calculate the percentage weight change for this interval
    weight_change = compute_weight_evolution(total_data)

    # insert the percentage weight change into the row
    total_row['Evolución de Peso (%)'] = weight_change

    # convert the new row into a DataFrame for concatenation
    total_row_df = pandas.DataFrame([total_row])

    # concatenate the new row with the phases DataFrame
    phases_df = pandas.concat([phases_df, total_row_df], ignore_index=True)

    return phases_df


def display_final_table(final_data):
    """
    Displays the final data table as an interactive plotly table.
    """
    # round numeric values to 2 decimal places
    rounded_data = final_data.round(2)

    # create table using plotly
    fig = plotly.graph_objects.Figure(data=[plotly.graph_objects.Table(
        header=dict(values=list(rounded_data.columns),
                    fill_color='paleturquoise',
                    align='center'),
        cells=dict(values=[rounded_data[col] for col in rounded_data.columns],
                   fill_color='lavender',
                   align='center'))
    ])

    fig.show()


def calculate_polynomial_regression(data, col, degree):
    x = numpy.arange(len(data['Fecha']))
    y = data[col].values

    # fit a polynomial of the given degree to the data
    coefficients = numpy.polyfit(x, y, degree)
    polynomial = numpy.poly1d(coefficients)
    return polynomial(x)


def plot_graphs(final_data):
    """
    Plots graphs for the specified columns in the final_data DataFrame.
    Formats the x-axis to show only the day and month (dd-mm).
    """
    # define the columns to plot and their colors
    columns_to_plot = ['Peso', 'Masa Muscular(kg)', 'Grasa corporal(%)', 'Grasa visceral', 'Calorías(kcal)',
                       'Proteínas(gr)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # convert the 'Fecha' column to datetime format
    final_data['Fecha'] = pandas.to_datetime(final_data['Fecha'], format='%Y/%m/%d')

    # create a subplot grid
    fig = plotly.subplots.make_subplots(rows=len(columns_to_plot), cols=1, shared_xaxes=False, vertical_spacing=0.05,
                           subplot_titles=columns_to_plot)

    # add a trace (line) for each column
    for i, (column, color) in enumerate(zip(columns_to_plot, colors), start=1):
        # plot data
        fig.add_trace(plotly.graph_objects.Scatter(
            x=final_data['Fecha'],
            y=final_data[column],
            mode='lines+markers',
            name=column,
            line=dict(color=color)
        ), row=i, col=1)

        # plot linear trend line
        degree = 1
        trend_line = calculate_polynomial_regression(final_data, column, degree=degree)
        fig.add_trace(plotly.graph_objects.Scatter(
            x=final_data['Fecha'],
            y=trend_line,
            mode='lines',
            name=f'Tendencia lineal',
            line=dict(color=color, dash='dash')
        ), row=i, col=1)

        # plot polynomial regression curve degree 2
        degree = 2
        polynomial = calculate_polynomial_regression(final_data, column, degree=degree)
        fig.add_trace(plotly.graph_objects.Scatter(
            x=final_data['Fecha'],
            y=polynomial,
            mode='lines',
            name=f'Tendencia polinómica (grado {degree})',
            line=dict(color=color, dash='longdashdot')
        ), row=i, col=1)

        # format the x-axis of each subplot
        fig.update_xaxes(
            tickformat="%d-%m",  # show day and month only
            dtick="D1",  # ensures that each day is a tick
            tickmode="linear",  # linear tick mode to ensure uniform spacing
            row=i, col=1
        )

    fig.update_layout(
        height=300 * len(columns_to_plot),  # adjust height according to the number of plots
        title_text='Valores recogidos a lo largo del periodo de déficit',
        showlegend=False
    )

    fig.show()


def plot_recent_weight_trend_old(data, days=7):
    """
    Plots the weight trend over the last 'days' days and shows the trend line.
    """
    recent_data = data.tail(days)
    seaborn.set(style="whitegrid")
    matplotlib.pyplot.figure(figsize=(10, 6))

    seaborn.lineplot(data=recent_data, x='Fecha', y='Peso', marker='o')
    matplotlib.pyplot.title(f'Weight Trend Over Last {days} Days')

    # trendline
    z = numpy.polyfit(range(len(recent_data['Peso'])), recent_data['Peso'], 1)
    p = numpy.poly1d(z)
    matplotlib.pyplot.plot(recent_data['Fecha'], p(range(len(recent_data['Peso']))), linestyle="--", color="red")

    # Calculate percentage change
    start_weight = recent_data['Peso'].iloc[0]
    end_weight = recent_data['Peso'].iloc[-1]
    weight_change = ((end_weight - start_weight) / start_weight) * 100
    matplotlib.pyplot.figtext(0.15, 0.8, f"Percentage Weight Change: {weight_change:.2f}%", fontsize=12)

    matplotlib.pyplot.show()


def plot_recent_weight_trend(final_data, days=7):
    """
    Plots weight data for the last N days along with linear and polynomial regression trends.
    Also calculates and displays the percentage change in weight over the period.
    """
    # convert 'Fecha' column to datetime format
    final_data['Fecha'] = pandas.to_datetime(final_data['Fecha'], format='%Y/%m/%d')

    # filter data for the last given days
    recent_data = final_data.tail(days)
    recent_data_for_nutrients = final_data.head(days)

    # calculate trends
    linear_trend = calculate_polynomial_regression(recent_data, 'Peso', degree=1)
    polynomial_trend = calculate_polynomial_regression(recent_data, 'Peso', degree=2)

    # calculate percentage change in weight
    percentage_change = compute_weight_evolution(recent_data)

    # calculate mean calories and protein consumed over the period
    mean_calories = recent_data_for_nutrients['Calorías(kcal)'].mean()
    mean_protein = recent_data_for_nutrients['Proteínas(gr)'].mean()
    disclaimer = "* Nota: El consumo de kcal y proteínas se calcula con los valores de cada día previo a los días del periodo mostrado."

    # get recommendation based on weight evolution
    recommendation = ''
    aiming_for_loss = True  # TODO: put this as a parameter when adding 'bulking' features
    if aiming_for_loss:
        recommended_weight_loss_interval = ((-0.5 * days) / 7, (-1 * days) / 7)
        if percentage_change > recommended_weight_loss_interval[0]:
            # not enough loss --> recommend larger deficit
            recommended_calories = (mean_calories - (mean_calories * 0.05), mean_calories - (mean_calories * 0.1))
            recommendation = f"Recomendación: reducir consumo de calorías al intervalo {recommended_calories: .2f}."
        elif percentage_change >= [1]:
            # sweet spot for body weight loss --> recommend no adjustment
            recommendation = f"Recomendación: mantener el mismo consumo de calorías."
        else:
            # body weight loss too high
            recommendation = f"Recomendación: pérdida de peso acelerada, aumentar ligeramente el consumo de calorías."

    # create a subplot grid
    fig = plotly.graph_objects.Figure()

    # add weight data
    fig.add_trace(plotly.graph_objects.Scatter(
        x=recent_data['Fecha'],
        y=recent_data['Peso'],
        mode='lines+markers',
        name='Peso',
        line=dict(color='#1f77b4')
    ))

    # add linear regression trend
    fig.add_trace(plotly.graph_objects.Scatter(
        x=recent_data['Fecha'],
        y=linear_trend,
        mode='lines',
        name='Tendencia lineal',
        line=dict(color='#ff7f0e', dash='dash')
    ))

    # add polynomial regression trend (degree 2)
    fig.add_trace(plotly.graph_objects.Scatter(
        x=recent_data['Fecha'],
        y=polynomial_trend,
        mode='lines',
        name='Tendencia polinómica (grado 2)',
        line=dict(color='#2ca02c', dash='longdashdot')
    ))

    # add an annotation for the mean calories and protein consumed
    fig.add_annotation(
        text=f"Media de kcal consumidas*: {mean_calories:.2f}",
        xref="paper", yref="paper",
        x=0.35, y=1.1, showarrow=False,
        font=dict(size=14)
    )
    fig.add_annotation(
        text=f"Media de proteínas consumidas*: {mean_protein:.2f}",
        xref="paper", yref="paper",
        x=0.35, y=1.05, showarrow=False,
        font=dict(size=14)
    )
    fig.add_annotation(
        text=recommendation,
        xref="paper", yref="paper",
        x=1, y=1.1, showarrow=False,
        font=dict(size=14)
    )
    fig.add_annotation(
        text=disclaimer,
        xref="paper", yref="paper",
        x=1, y=1.05, showarrow=False,
        font=dict(size=14)
    )

    # update layout and x-axis formatting
    fig.update_layout(
        title=f'Evolución peso últimos {days} días ({percentage_change:.2f}%)',
        xaxis_title='Fecha',
        yaxis_title='Peso (kg)',
        xaxis=dict(
            tickformat="%d-%m",  # show day and month only
            dtick="D1",  # ensures that each day is a tick
            tickmode="linear"  # linear tick mode to ensure uniform spacing
        )
    )

    fig.show()


def main():
    # input parameters
    file_path = '../data/fitdays.csv'
    run_full_analysis_option = input(
        'Introduce una de las siguientes opciones:\n'
        '\t1. Ejecutar análisis completo de déficit.\n'
        '\t2. Mostrar tendencia de peso de los últimos N días.\n'
        '(1/2): ')

    if run_full_analysis_option == '1':
        maintenance_start_date_input = input('Introduce la fecha de inicio de la fase de mantenimiento (DD-MM-YYYY): ')
        list_of_numbers_maintenance_start = maintenance_start_date_input.split('-')
        deficit_end_date = input('Introduce la fecha de fin del déficit (DD-MM-YYYY): ')
        list_of_numbers_deficit_end = deficit_end_date.split('-')
        maintenance_start_date = pandas.to_datetime(
            str(list_of_numbers_maintenance_start[2]) + '-' + str(list_of_numbers_maintenance_start[1]) + '-' + str(
                list_of_numbers_maintenance_start[0]))
        deficit_end_date = pandas.to_datetime(
            str(list_of_numbers_deficit_end[2]) + '-' + str(list_of_numbers_deficit_end[1]) + '-' + str(
                list_of_numbers_deficit_end[0]))
        maintenance_duration = int(input('Número de días de duración de la fase de mantenimiento (default=10): '))
        pre_deficit_duration = int(input('Número de días de duración de la fase pre-déficit (default=3): '))
        phases_duration = int(input('Número de días de duración de las fases de déficit (default=7): '))

        # main process
        data = read_data(file_path)
        filtered_data = filter_data_by_date(data, maintenance_start_date, deficit_end_date)
        phases = split_and_aggregate_phases(filtered_data, maintenance_duration, pre_deficit_duration, phases_duration)
        phases = calculate_weight_percentage_change_per_phase(filtered_data, phases)

        start_deficit_date = maintenance_start_date + timedelta(days=maintenance_duration + pre_deficit_duration)
        final_data = calculate_total_evolution(filtered_data, phases, start_deficit_date)

        display_final_table(final_data)

        # plots
        plot_graphs(filtered_data)
        plot_recent_weight_trend(filtered_data, phases_duration)

    elif run_full_analysis_option == '2':
        num_days_to_plot = int(input('Introduce el número de días para mostrar tendencia (min=2): '))

        data = read_data(file_path)
        end_date = data['Fecha'].max()
        start_date = end_date - timedelta(days=num_days_to_plot)
        filtered_data = filter_data_by_date(data, start_date, end_date)
        display_final_table(filtered_data)
        plot_recent_weight_trend(filtered_data, num_days_to_plot)

    else:
        print('Opción no válida.')

    print('End!')


if __name__ == "__main__":
    main()
