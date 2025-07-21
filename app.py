from shiny import App, render, ui, reactive, Session
from shinywidgets import output_widget, render_widget  
from shiny.ui import HTML
import shinyswatch
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

from plotnine import *
from plotnine import aes, ggplot
from plotnine.geoms import geom_col, geom_line
from plotnine.geoms.geom_point import geom_point
from plotnine.geoms.geom_smooth import geom_smooth
import plotly.express as px
import plotly.graph_objects as go
from plotly.callbacks import Points

from data_manipulation.data_wrangling import summary_by_time, summary_by_time_weekly, anamoly_detection
from datetime import datetime, timedelta

import langchain
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import tabulate
import json
from pandas import json_normalize
import re
import os
from uuid import uuid4
import ibis
import duckdb




analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])

summary_data = summary_by_time(analysis_data,
                               'Site_Id','Parameter_ParameterDescription')


summary_data_weekly = summary_by_time_weekly(analysis_data,
                               'Site_Id','Parameter_ParameterDescription')

anomaly_data = anamoly_detection(summary_data_weekly, 
                                'Site_Id','Parameter_ParameterDescription')

summary_data['Site_Id'] = summary_data['Site_Id'].astype(str)

# change the name of Value_mean to PM 10 values

summary_data.rename(columns = {'Value_mean': 'PM 10'},
                     inplace = True)       

#-----------------------------------------------------------
# time slicers

today = datetime.now()
thirteen_weeks_ago = today - timedelta(weeks = 13)

year_ago = today - timedelta(days = 365)
thirteen_weeks_ago_last_year = year_ago - timedelta(weeks = 13)

two_years_ago = today - timedelta(days = 730)

# Get the current year
current_year = datetime.now().year

# Get the last year

last_year = year_ago.year

# Create a datetime object for the start of the year
this_year_start = datetime(current_year, 1, 1)

last_year_start = datetime(last_year, 1, 1)


#--------------------------------------------------------------

# OPENAI SETUP
model = 'gpt-4o-mini'

# Initialize the LLM
llm = ChatOpenAI(
    model_name=model,
    temperature=0,
    api_key =  os.getenv("OPENAI_API_KEY")
)




#--------------------------------------------------------------




app_ui = ui.page_fluid(
        #  ui.tags.style(HTML(custom_css)),
            # ui.tags.style(HTML(custom_css)),
             ui.page_navbar(title = "Pollution Pulse"),
             ui.navset_pill(
                 ui.nav_panel("Comprative Analysis of Air Pollution",
                              ui.layout_sidebar(
                                  ui.sidebar( 
                                              ui.input_radio_buttons(
                                            "var", 
                                            "Select a Site ID",
                                              choices= list(summary_data['Site_Id']),
                                            selected = '329',
                                                                    ),
                                            ui.input_radio_buttons(
                                            "pollutant", 
                                            "Select a pollutant",
                                              choices= list(summary_data['Parameter_ParameterDescription']),
                                            selected = 'PM10',
                                                                    )                       
                                            ),
                                           
                                             ui.layout_columns(
                                                             
                                                                ui.row(ui.output_data_frame("insights_1"),
                                                                
                                                                                        ),
                                                                       
                                                               
                                                                ui.row(ui.input_radio_buttons("time",
                                                                "Select a time period",
                                                                choices = ["This Quarter",
                                                                           "Last 52 Weeks",
                                                                           "Year to date"],
                                                                           selected='This Quarter',
                                                                           inline=True),
                                                                ui.card(output_widget("line_plot"),
                                                                        full_screen=True)
                                                             
                                                                        ),
                                                                         ui.input_action_button("clear_lines",
                                                                                                "Clear Data"),
                                                                          col_widths=[4,7,1],
                                                                         ),
                                         ui.layout_columns(
                                                                    
                                                                 ui.output_data_frame("insights_2"),
                                                                ui.row(ui.input_radio_buttons("time_comparative",
                                                                "Select a comparative time period",
                                                                choices = ["Last Quarter",
                                                                           "Previous 52 Weeks",
                                                                           "Previous year to date"],
                                                                           selected = 'Last Quarter',
                                                                           inline=True),
                                         ui.card(output_widget("line_plot_comparative"),full_screen=True)),
                                                                           ui.input_action_button("clear_lines_comparative",
                                                                                                "Clear Data"),
                                                                            col_widths=[4,7,1]
                                                            ),
                                                )
                             )
                                  ,ui.nav_panel("Anomaly Detection",
                                                ui.layout_sidebar(
                                                    ui.sidebar(ui.input_radio_buttons(
                                                                "var_2", 
                                                                "Select a Site ID",
                                                                choices= list(summary_data['Site_Id']),
                                                                selected = '329',
                                                                                        ),
                                                                ui.input_radio_buttons(
                                                                "pollutant_anomaly", 
                                                                "Select a pollutant",
                                                                choices= list(summary_data['Parameter_ParameterDescription']),
                                                                selected = 'PM10',
                                                                    )    ),
                                                    output_widget("anomaly_plot")
                                                )))
                                  
                                 ,
                                   theme = shinyswatch.theme.minty(),
                      )
 


# Server function provides access to client-side input values
def server(input, output, session: Session):
    # Store multiple pinned points with positions
    # pinned_points = reactive.value([])

    click_reactive = reactive.value() 
    hover_reactive = reactive.value() 
    selection_reactive = reactive.value() 

    # Store the current filtered dataset in a reactive value
    current_dataset = reactive.value(pd.DataFrame())

    
    points_store = reactive.value([])
    plot_counter = reactive.value(0)

    points_store_comparative = reactive.value([])
    plot_counter_comparative = reactive.value(0)

    def create_plot(f):

  
        fig  = go.Figure()

        fig.add_trace(go.Scatter(
             x = f['Date'],
             y = f['PM 10'],
            
             fill = 'tozeroy',
             mode = 'lines + markers',
           
             fillcolor= 'rgba(0, 0, 255, 0.5)',
             line = dict(color = '#118DFF'),
             showlegend = False,
             hoverinfo='x+y'
        ))

        y_min_primary = f['PM 10'].min() -2
        y_max_primary = f['PM 10'].max() + 2

        for i, point in enumerate(points_store()):
            date_val, y1_val = point
            color = f'hsl({(i * 60) % 360}, 70%, 50%)'

            #Add vertical line at the date value
            fig.add_trace(go.Scatter(
                x = [date_val,date_val],
                y = [y_min_primary,y_max_primary],
                mode = 'lines',
                line = dict(color = 'blue',
                            width = 2,
                            dash = 'dash',
                            
                            ), 
                         
                            showlegend = False


                                    )
                                    )
            


        
            annotation_text = (
                f"Date: {date_val}<br>"
                f"Pollutant value: {round(y1_val,2)}<br>"
            )


            fig.add_annotation(
                x = date_val,
                y = y1_val + 5,
                xref = "x",
                yref="y",
                text = annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                bgcolor='white',
                bordercolor='red',

            )


        # Update marker styles
        fig.update_layout(
                           
                           
                            template = 'plotly_white'
                                        )
        fig.update_layout(title_font_family="Times New Roman",
                                        title_font_color='#118DFF',
                                        title = {'x':0.5,
                                                'xanchor':'center'})
    

     

        return fig



    def create_comparative_plot(f):

  
        fig  = go.Figure()

        fig.add_trace(go.Scatter(
             x = f['Date'],
             y = f['PM 10'],
             fill = 'tozeroy',
             mode = 'lines + markers',
           
             fillcolor= 'rgba(163, 221, 179, 0.5)',
              line = dict(color = '#A3DDB3'),
             showlegend = False,
             hoverinfo='x+y'
        ))

        y_min_primary = f['PM 10'].min() -2
        y_max_primary = f['PM 10'].max() + 2

        for i, point in enumerate(points_store_comparative()):
            date_val, y1_val = point
            color = f'hsl({(i * 60) % 360}, 70%, 50%)'

            #Add vertical line at the date value
            fig.add_trace(go.Scatter(
                x = [date_val,date_val],
                y = [y_min_primary,y_max_primary],
                mode = 'lines',
                line = dict(color = 'green',
                            width = 2,
                            dash = 'dash',
                            
                            ), 
                         
                            showlegend = False


                                    )
                                    )
            


        
            annotation_text = (
                f"Date: {date_val}<br>"
                f"Pollutant value: {round(y1_val,2)}<br>"
            )


            fig.add_annotation(
                x = date_val,
                y = y1_val + 5,
                xref = "x",
                yref="y",
                text = annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                bgcolor='white',
                bordercolor='red',

            )


        # Update marker styles
        fig.update_layout(
                            
                           
                                        template = 'plotly_white'
                                        )
        fig.update_layout(title_font_family="Times New Roman",
                                        title_font_color='#A3DDB3',
                                        title = {'x':0.5,
                                                'xanchor':'center'})
    

     

        return fig


    @render_widget
    def line_plot():

        plot_counter()
        
        d = summary_data[summary_data['Site_Id'] == input.var()]

        d = d[d['Parameter_ParameterDescription'] == input.pollutant()]

       

        

        if input.time() == 'This Quarter':
          
            filtered_data = d[(d['Date']> thirteen_weeks_ago)
                                & (d['Date'] < today)]
            
            filtered_data["Date"] = filtered_data["Date"].dt.strftime('%Y-%m-%d')


            fig = create_plot(filtered_data)

            widget = go.FigureWidget(fig)


            def handle_click(trace, points, selector):

                if len(points.point_inds)>0:
                    idx = points.point_inds[0]
                    date_val = filtered_data['Date'].iloc[idx]

                    y1_val = filtered_data['PM 10'].iloc[idx]

                    # update the reactive value with the new point

                    current_points = points_store()

                    current_points.append((date_val,
                                        y1_val))
                    
                    points_store.set(current_points)

                    # Force a redraw by increneting the counter

                    plot_counter.set(plot_counter() + 1)

            widget.data[0].on_click(handle_click)

            return widget
        
        if input.time() == 'Last 52 Weeks':
          
            filtered_data = d[(d['Date']> year_ago)
                                & (d['Date'] < today)]
            
            filtered_data["Date"] = filtered_data["Date"].dt.strftime('%Y-%m-%d')


            fig = create_plot(filtered_data)

            widget = go.FigureWidget(fig)


            def handle_click_2(trace, points, selector):

                if len(points.point_inds)>0:
                    idx = points.point_inds[0]
                    date_val = filtered_data['Date'].iloc[idx]

                    y1_val = filtered_data['PM 10'].iloc[idx]

                    # update the reactive value with the new point

                    current_points = points_store()

                    current_points.append((date_val,
                                        y1_val))
                    
                    points_store.set(current_points)

                    # Force a redraw by increneting the counter

                    plot_counter.set(plot_counter() + 1)

            widget.data[0].on_click(handle_click_2)

            return widget
        
        if input.time() == 'Year to date':

            filtered_data = d[(d['Date']> this_year_start)
                             & (d['Date'] < today)]
            
            filtered_data["Date"] = filtered_data["Date"].dt.strftime('%Y-%m-%d')

            
            fig = create_plot(filtered_data)

            widget = go.FigureWidget(fig)


            def handle_click_3(trace, points, selector):

                if len(points.point_inds)>0:
                    idx = points.point_inds[0]
                    date_val = filtered_data['Date'].iloc[idx]

                    y1_val = filtered_data['PM 10'].iloc[idx]

                    # update the reactive value with the new point

                    current_points = points_store()

                    current_points.append((date_val,
                                        y1_val))
                    
                    points_store.set(current_points)

                    # Force a redraw by increneting the counter

                    plot_counter.set(plot_counter() + 1)

            widget.data[0].on_click(handle_click_3)

            return widget
     

    @reactive.effect
    @reactive.event(input.clear_lines)
    def clear_reference_lines():

        points_store.set([])

        plot_counter.set(plot_counter() + 1)

    
    @reactive.effect
    @reactive.event(input.clear_lines_comparative)
    def clear_reference_lines_comparative():

        points_store_comparative.set([])

        plot_counter_comparative.set(plot_counter_comparative() + 1)


    
    @render_widget  
    def line_plot_comparative():
        
        
        plot_counter_comparative()
        # d = summary_data
        d = summary_data[summary_data['Site_Id'] == input.var()]

        d = d[d['Parameter_ParameterDescription'] == input.pollutant()]

        if input.time_comparative() == "Last Quarter":
           
            filtered_data = d[(d['Date']> thirteen_weeks_ago_last_year)
                             & (d['Date'] < year_ago)]
            
            filtered_data["Date"] = filtered_data["Date"].dt.strftime('%Y-%m-%d')

            fig = create_comparative_plot(filtered_data)

            widget = go.FigureWidget(fig)


            def handle_click_4(trace, points, selector):

                if len(points.point_inds)>0:
                    idx = points.point_inds[0]
                    date_val = filtered_data['Date'].iloc[idx]

                    y1_val = filtered_data['PM 10'].iloc[idx]

                    # update the reactive value with the new point

                    current_points = points_store_comparative()

                    current_points.append((date_val,
                                        y1_val))
                    
                    points_store_comparative.set(current_points)

                    # Force a redraw by increneting the counter

                    plot_counter_comparative.set(plot_counter_comparative() + 1)

            widget.data[0].on_click(handle_click_4)

            return widget

        if input.time_comparative() == "Previous 52 Weeks":
    
            filtered_data = d[(d['Date']> two_years_ago)
                             & (d['Date'] < year_ago)]
            
            filtered_data["Date"] = filtered_data["Date"].dt.strftime('%Y-%m-%d')

            fig = create_comparative_plot(filtered_data)

            widget = go.FigureWidget(fig)


            def handle_click_5(trace, points, selector):

                if len(points.point_inds)>0:
                    idx = points.point_inds[0]
                    date_val = filtered_data['Date'].iloc[idx]

                    y1_val = filtered_data['PM 10'].iloc[idx]

                    # update the reactive value with the new point

                    current_points = points_store_comparative()

                    current_points.append((date_val,
                                        y1_val))
                    
                    points_store_comparative.set(current_points)

                    # Force a redraw by increneting the counter

                    plot_counter_comparative.set(plot_counter_comparative() + 1)

            widget.data[0].on_click(handle_click_5)

            return widget

            # pollution_plot = px.area(data_frame=d,
            #                             x="Date",
            #                                 y="PM 10",
            #                                 markers = True,
                                            
            #                                 color_discrete_sequence=['#A3DDB3'],
            #                                 template='plotly_white',
            #                                 title = f"Analysing previous 52 weeks: {two_years_ago.date()} to {year_ago.date()}"
            #                                 )
            
            # # Update marker styles
            # pollution_plot.update_traces(mode='lines+markers',
            #                               marker=dict(size=6, symbol='diamond', 
            #                              color='black'))
            
            # pollution_plot.update_layout(title_font_family="Times New Roman",
            #                              title_font_color='#A3DDB3',
            #                              title = {'x':0.5,
            #                                       'xanchor':'center'})
            

        if input.time_comparative() == "Previous year to date":
         
            filtered_data = d[(d['Date']> last_year_start)
                             & (d['Date'] < year_ago)]
            
            filtered_data["Date"] = filtered_data["Date"].dt.strftime('%Y-%m-%d')

            fig = create_comparative_plot(filtered_data)

            widget = go.FigureWidget(fig)


            def handle_click_6(trace, points, selector):

                if len(points.point_inds)>0:
                    idx = points.point_inds[0]
                    date_val = filtered_data['Date'].iloc[idx]

                    y1_val = filtered_data['PM 10'].iloc[idx]

                    # update the reactive value with the new point

                    current_points = points_store_comparative()

                    current_points.append((date_val,
                                        y1_val))
                    
                    points_store_comparative.set(current_points)

                    # Force a redraw by increneting the counter

                    plot_counter_comparative.set(plot_counter_comparative() + 1)

            widget.data[0].on_click(handle_click_6)

            return widget

        #     pollution_plot = px.area(data_frame=d,
        #                                 x="Date",
        #                                     y="PM 10",
        #                                     markers = True,
                                            
        #                                     color_discrete_sequence=['#A3DDB3'],
        #                                     template='plotly_white',
        #                                     title = f"Analysing previous year to date: {last_year_start.date()} to {year_ago.date()}"
        #                                     )
            
        #     # Update marker styles
        #     pollution_plot.update_traces(mode='lines+markers',
        #                                   marker=dict(size=6, symbol='diamond', 
        #                                  color='black'))
            
        #     pollution_plot.update_layout(title_font_family="Times New Roman",
        #                                  title_font_color='#A3DDB3',
        #                                  title = {'x':0.5,
        #                                           'xanchor':'center'})


        # return pollution_plot
    
    @output
    @render.data_frame
    def insights_1():

        d = summary_data[summary_data['Site_Id'] == input.var()]
        d = d[d['Parameter_ParameterDescription'] == input.pollutant()]

        if input.time() == "This Quarter":
           
            d = d[(d['Date']> thirteen_weeks_ago)
                             & (d['Date'] < today)]
            
            # Create an agent that can interact with the Pandas DataFrame
            data_analysis_agent = create_pandas_dataframe_agent(
                llm, 
                d, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                suffix="Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True
            )

            response = data_analysis_agent.invoke("Can you summarize the data and highlight key findings in 2 senstences?")

           
            response_output_df = json_normalize(response)

            response_output_df.rename(columns={'output':"key insights for this quarter"},
                                       inplace = True)
            
            response_output_df = response_output_df.drop(columns='input')
            
           

            return render.DataTable(response_output_df,
                                    editable = True,
                                    width = "700px",
                                    
                                    
                                    )
        
        if input.time() == "Last 52 Weeks":
           
            d = d[(d['Date']> year_ago)
                             & (d['Date'] < today)]
            
            # Create an agent that can interact with the Pandas DataFrame
            data_analysis_agent = create_pandas_dataframe_agent(
                llm, 
                d, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                suffix="Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True
            )

            response = data_analysis_agent.invoke("Can you summarize the data and highlight key findings in 2 senstences?")

           
            response_output_df = json_normalize(response)

            response_output_df.rename(columns={'output':"key insights for last 52 weeks"},
                                       inplace = True)
            
            response_output_df = response_output_df.drop(columns='input')
            
           

            return render.DataTable(response_output_df,
                                    editable = True,
                                    width = "700px",
                                    )
        
        if input.time() == "Year to date":
           
            d = d[(d['Date']> this_year_start)
                             & (d['Date'] < today)]
            
            
            
            # Create an agent that can interact with the Pandas DataFrame
            data_analysis_agent = create_pandas_dataframe_agent(
                llm, 
                d, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                suffix="Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True
            )

            response = data_analysis_agent.invoke("Can you summarize the data and highlight key findings in 2 senstences?")

           
            response_output_df = json_normalize(response)

            response_output_df.rename(columns={'output':"key insights for year to date"},
                                       inplace = True)
            
            response_output_df = response_output_df.drop(columns='input')

          
            
            return render.DataTable(response_output_df,
                                    editable = True,
                                    width = "700px",
                                  )
    
    @output
    @render.data_frame
    def insights_2():

        d = summary_data[summary_data['Site_Id'] == input.var()]

        d = d[d['Parameter_ParameterDescription'] == input.pollutant()]

        if input.time_comparative() == "Last Quarter":
           
            d = d[(d['Date']> thirteen_weeks_ago_last_year)
                             & (d['Date'] < year_ago)]
            
            # Create an agent that can interact with the Pandas DataFrame
            data_analysis_agent = create_pandas_dataframe_agent(
                llm, 
                d, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                suffix="Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True
            )

            response = data_analysis_agent.invoke("Can you summarize the data and highlight key findings in 2 senstences?")

           
            response_output_df = json_normalize(response)

            response_output_df.rename(columns={'output':"key insights for last quarter"},
                                       inplace = True)
            
            response_output_df = response_output_df.drop(columns='input')

            

            return render.DataTable(response_output_df,
                                    editable = True,
                                    width = "700px",
                                    )
        
        if input.time_comparative() == "Previous 52 Weeks":
           
            d = d[(d['Date']> two_years_ago)
                             & (d['Date'] < year_ago)]
            
            # Create an agent that can interact with the Pandas DataFrame
            data_analysis_agent = create_pandas_dataframe_agent(
                llm, 
                d, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                suffix="Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True
            )

            response = data_analysis_agent.invoke("Can you summarize the data and highlight key findings in 2 senstences?")

           
            response_output_df = json_normalize(response)

            response_output_df.rename(columns={'output':"key insights for previous 52 weeks"},
                                       inplace = True)
            response_output_df = response_output_df.drop(columns='input')
           
            
            return render.DataTable(response_output_df,
                                    editable = True,
                                    width = "700px",
                                   )
        
        if input.time_comparative() == "Previous year to date":
           
            d = d[(d['Date']> last_year_start)
                             & (d['Date'] < year_ago)]
            
            # Create an agent that can interact with the Pandas DataFrame
            data_analysis_agent = create_pandas_dataframe_agent(
                llm, 
                d, 
                agent_type=AgentType.OPENAI_FUNCTIONS,
                suffix="Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True
            )

            response = data_analysis_agent.invoke("Can you summarize the data and highlight key findings in 2 senstences?")

           
            response_output_df = json_normalize(response)

            response_output_df.rename(columns={'output':"key insights for previous year to date"},
                                       inplace = True)
            
            response_output_df = response_output_df.drop(columns='input')

           
            return render.DataTable(response_output_df,
                                    editable = True,
                                    width = "700px",
                                   )
        
    @render_widget
    def anomaly_plot():
        
        # anomaly_data_filtered = anomaly_data[anomaly_data['Site_Id'] == input.var_2()]

        anomaly_data_filtered = anomaly_data[anomaly_data['Parameter_ParameterDescription'] == input.pollutant_anomaly()]
        fig =  anomaly_data_filtered \
                .groupby(["Site_Id"]) \
                .plot_anomalies(
                    date_column = "Date", 
                    facet_ncol = 2, 
                    width = 2000,
                    height = 1000,
                  
                  
                )
        
        return fig

    
app = App(app_ui, server)

