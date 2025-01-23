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


analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])

summary_data = summary_by_time(analysis_data,
                               'Site_Id')

summary_data['Site_Id'] = summary_data['Site_Id'].astype(str)

#-----------------------------------------------------------
# time slicers

today = datetime.now()
thirteen_weeks_ago = today - timedelta(weeks = 13)

year_ago = today - timedelta(days = 365)
thirteen_weeks_ago_last_year = year_ago - timedelta(weeks = 13)

two_years_ago = today - timedelta(days = 730)

this_year_start = '2025-01-01'
last_year_start = '2024-01-01'


app_ui = ui.page_fluid(
      
             ui.page_navbar(title = "Pollution Pulse"),
             ui.navset_card_pill(
                 ui.nav_panel("Air Pollution Analysis",
                              ui.layout_sidebar(
                                  ui.sidebar( ui.input_radio_buttons(
                                            "var", 
                                            "Select a Site ID",
                                              choices= list(summary_data['Site_Id']),
                                            selected = '329',
                                                                    )
                                            ),
                                           
                                             ui.layout_columns(ui.output_text("insights_1"),
                                                                   ui.row(ui.input_radio_buttons("time",
                                                                "Select a time period",
                                                                choices = ["This Quarter",
                                                                           "Last 52 Weeks",
                                                                           "YTD"],
                                                                           selected='This Quarter',
                                                                           inline=True),
                                         output_widget("line_plot", width = '120%')), col_widths=[4,8]),
                                         ui.layout_columns(ui.output_text("insights_2"),
                                                           ui.row(ui.input_radio_buttons("time_comparative",
                                                                "Select a time period",
                                                                choices = ["Last Quarter",
                                                                           "Previous 52 Weeks",
                                                                           "YTD-1"],
                                                                           selected = 'Last Quarter',
                                                                           inline=True),
                                         output_widget("line_plot_comparative")),col_widths=[4,8]) 
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

        if input.time() == "This Quarter":
           
            d = d[(d['Date']> thirteen_weeks_ago)
                             & (d['Date'] < today)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="Value_mean",
                                            markers = True,
                                           color_discrete_sequence=['#118DFF'],
                                           template='plotly_white'
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))
      
        
        if input.time() == "Last 52 Weeks":
            # d = summary_data[summary_data['Site_Id'] == input.var()]
            d = d[(d['Date']> year_ago)
                             & (d['Date'] < today)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="Value_mean",
                                            markers = True,
                                           color_discrete_sequence=['#118DFF'],
                                           template='plotly_white')
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))

        return pollution_plot
    
    @render_widget  
    def line_plot_comparative():
        
        # d = summary_data
        d = summary_data[summary_data['Site_Id'] == input.var()]

        if input.time_comparative() == "Last Quarter":
           
            d = d[(d['Date']> thirteen_weeks_ago_last_year)
                             & (d['Date'] < year_ago)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="Value_mean",
                                            markers = True,
                                            color_discrete_sequence=['#A3DDB3'],
                                            template='plotly_white'
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))

        
        if input.time_comparative() == "Previous 52 Weeks":
            # d = summary_data[summary_data['Site_Id'] == input.var()]
            d = d[(d['Date']> two_years_ago)
                             & (d['Date'] < year_ago)]
            
            pollution_plot = px.area(data_frame=d,
                                        x="Date",
                                            y="Value_mean",
                                            markers = True,
                                            
                                            color_discrete_sequence=['#A3DDB3'],
                                            template='plotly_white'
                                            )
            
            # Update marker styles
            pollution_plot.update_traces(mode='lines+markers',
                                          marker=dict(size=6, symbol='diamond', 
                                         color='black'))


        return pollution_plot
    
    @render.text
    def insights_1():

        return "place holder for insights for the selected time"
    
    @render.text
    def insights_2():

        return "place holder for insights for the selected time"

    
app = App(app_ui, server)

