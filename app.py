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

from data_manipulation.data_wrangling import summary_by_time
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


analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])

summary_data = summary_by_time(analysis_data,
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
    
            # ui.tags.style(HTML(custom_css)),
             ui.page_navbar(title = "Pollution Pulse"),
             ui.navset_pill(
                 ui.nav_panel("Comprative Analysis of Air Pollution",
                              ui.layout_sidebar(
                                  ui.sidebar( ui.input_radio_buttons(
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
                                                             
                                                                ui.output_data_frame("insights_1"),
                                                                       
                                                               
                                                                ui.row(ui.input_radio_buttons("time",
                                                                "Select a time period",
                                                                choices = ["This Quarter",
                                                                           "Last 52 Weeks",
                                                                           "Year to date"],
                                                                           selected='This Quarter',
                                                                           inline=True),
                                         ui.card(output_widget("line_plot"),full_screen=True)), col_widths=[4,8]),
                                         ui.layout_columns(
                                                                    
                                                                 ui.output_data_frame("insights_2"),
                                                                ui.row(ui.input_radio_buttons("time_comparative",
                                                                "Select a comparative time period",
                                                                choices = ["Last Quarter",
                                                                           "Previous 52 Weeks",
                                                                           "Previous year to date"],
                                                                           selected = 'Last Quarter',
                                                                           inline=True),
                                         ui.card(output_widget("line_plot_comparative"),full_screen=True)),col_widths=[4,8]) 
                                                )
                             )
                                 ), theme = shinyswatch.theme.minty(),
                      )
 


# Server function provides access to client-side input values
def server(input, output, session: Session):

    @render_widget  
    def line_plot():
        
        # d = summary_data
        d = summary_data[summary_data['Site_Id'] == input.var()]

        d = d[d['Parameter_ParameterDescription'] == input.pollutant()]

        if input.time() == "This Quarter":
           
            d = d[(d['Date']> thirteen_weeks_ago)
                             & (d['Date'] < today)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="PM 10",
                                            markers = True,
                                           color_discrete_sequence=['#118DFF'],
                                           template='plotly_white',
                                           title = f"Analysing this quarter: {thirteen_weeks_ago.date()} to {today.date()}"
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
            pollution_plot.update_layout(title_font_family="Times New Roman",
                                         title_font_color='#118DFF',
                                         title = {'x':0.5,
                                                  'xanchor':'center'})
      
        
        if input.time() == "Last 52 Weeks":
           
            d = d[(d['Date']> year_ago)
                             & (d['Date'] < today)]
            
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="PM 10",
                                            markers = True,
                                           color_discrete_sequence=['#118DFF'],
                                           template='plotly_white',
                                           title = f"Analysing the last 52 weeks:{year_ago.date()} to {today.date()}")
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
            
            pollution_plot.update_layout(title_font_family="Times New Roman",
                                         title_font_color='#118DFF',
                                         title = {'x':0.5,
                                                  'xanchor':'center'})
            
        if input.time() == "Year to date":
          
            d = d[(d['Date']> this_year_start)
                             & (d['Date'] < today)]
            
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="PM 10",
                                            markers = True,
                                           color_discrete_sequence=['#118DFF'],
                                           template='plotly_white',
                                           title = f"Analysing year to date: {this_year_start.date()} to {today.date()}")
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
            
            pollution_plot.update_layout(title_font_family="Times New Roman",
                                         title_font_color='#118DFF',
                                         title = {'x':0.5,
                                                  'xanchor':'center'})

        return pollution_plot
    
    @render_widget  
    def line_plot_comparative():
        
        # d = summary_data
        d = summary_data[summary_data['Site_Id'] == input.var()]

        d = d[d['Parameter_ParameterDescription'] == input.pollutant()]

        if input.time_comparative() == "Last Quarter":
           
            d = d[(d['Date']> thirteen_weeks_ago_last_year)
                             & (d['Date'] < year_ago)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="PM 10",
                                            markers = True,
                                            color_discrete_sequence=['#A3DDB3'],
                                            template='plotly_white',
                                            title = f"Analysing Last Quarter: {thirteen_weeks_ago_last_year.date()} to {year_ago.date()}"
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
            
            pollution_plot.update_layout(title_font_family="Times New Roman",
                                         title_font_color='#A3DDB3',
                                         title = {'x':0.5,
                                                  'xanchor':'center'})

        
        if input.time_comparative() == "Previous 52 Weeks":
    
            d = d[(d['Date']> two_years_ago)
                             & (d['Date'] < year_ago)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="PM 10",
                                            markers = True,
                                            
                                            color_discrete_sequence=['#A3DDB3'],
                                            template='plotly_white',
                                            title = f"Analysing previous 52 weeks: {two_years_ago.date()} to {year_ago.date()}"
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
            
            pollution_plot.update_layout(title_font_family="Times New Roman",
                                         title_font_color='#A3DDB3',
                                         title = {'x':0.5,
                                                  'xanchor':'center'})
            

        if input.time_comparative() == "Previous year to date":
         
            d = d[(d['Date']> last_year_start)
                             & (d['Date'] < year_ago)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="PM 10",
                                            markers = True,
                                            
                                            color_discrete_sequence=['#A3DDB3'],
                                            template='plotly_white',
                                            title = f"Analysing previous year to date: {last_year_start.date()} to {year_ago.date()}"
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
            
            pollution_plot.update_layout(title_font_family="Times New Roman",
                                         title_font_color='#A3DDB3',
                                         title = {'x':0.5,
                                                  'xanchor':'center'})


        return pollution_plot
    
    @output
    @render.data_frame
    def insights_1():

        d = summary_data[summary_data['Site_Id'] == input.var()]

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
        

    
app = App(app_ui, server)

