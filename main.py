#!/usr/bin/env python3
import pandas as pd
import numpy as np

from tkinter import *
from DataClean import *
import matplotlib.pyplot as plt
import plotly.express as px

# Instantiate root window
#def run():
    #print (mb.get())

# Read csv files
vacdata = pd.read_csv("/Users/mawi/Documents/data_vaccinations/country_vaccinations.csv")
popdata = pd.read_csv("/Users/mawi/Documents/population_by_country_2020.csv")
#numpyvacdata = np.genfromtxt("/Users/mawi/Documents/data_vaccinations/country_vaccinations.csv",
                             #names=True, delimiter=",", usecols=range(14)).reshape(14,-1)


#cleanup

# Rename columns in population dataset
popdata_new = popdata.rename(columns={'Country (or dependency)': 'country', 'Population (2020)': 'population'}, inplace=False)

# Set missing values to 0
#vacdata['people_fully_vaccinated'] = vacdata['people_fully_vaccinated'].fillna(0)

#vacdata['people_fully_vaccinated'][3171] = 10

# Try setting Denmarks first value to 0
#vacdata.iat[3171, 5] =0

#for col in vacdata.columns:
#      print(col)


# set missing values to previous values
#vacdata['people_fully_vaccinated'].fillna(method='pad', inplace=True)


clean_data_vac = DataClean(vacdata)
clean_data_pop = DataClean(popdata_new)

# Drops Items in dropList
dropListVac = ['iso_code', 'total_vaccinations', 'people_vaccinated', 'daily_vaccinations_raw',
               'daily_vaccinations', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
               'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million', 'vaccines',
               'source_name', 'source_website']
dropListPop = ['Yearly Change', 'Net Change', 'Density (P/Km²)', 'Land Area (Km²)', 'Migrants (net)',
               'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share']

clean_data_vac.removeCols(dropListVac)
clean_data_pop.removeCols(dropListPop)

# Prep for dropdown menu
countries = popdata_new.country.to_list()
sorted_pop = sorted(countries)

# Group data
people_fully_vaccinated = vacdata.groupby(by=['country'], sort=False, as_index=False)['people_fully_vaccinated'].max()

#how many countries are
#print("grouped data:")
#print(people_fully_vaccinated.head(10))

#print("vaccination data:")
#print(vacdata.head(10))
#print("population data:")
#print(popdata_new.head(10))

#clean_data_pop.combineDataSets(people_fully_vaccinated)


# GUI
root = Tk()
root.title("Corona vaccination prediction")

# Add a grid
mainframe = Frame(root, width=300, height=200)
mainframe.grid(row=0, column=0)
mainframe.columnconfigure(0, weight = 3)
mainframe.columnconfigure(1, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.rowconfigure(1, weight = 3)
#mainframe.pack(pady = 100, padx = 100)
#mainframe.pack(side=TOP, expand=NO, fill=NONE)

# Create a Tkinter variable
tkvar = StringVar(root)
# Set default option
tkvar.set("Choose country")


message = Label(mainframe, text="Pick a country below, and we'll predict when it will be fully vaccinated.")
mb = OptionMenu(mainframe, tkvar, *sorted_pop)
#mb.grid(row=3, column=0)

message.grid(row=1, column=0)
mb.grid(row=4, column=0)

# Make the value change
def change_dropdown(*args):
    print(tkvar.get())

# Link function to change dropdown
#tkvar.trace('w', change_dropdown())

predictbutton = Button(mainframe, text="Predict!").grid(row=6, column=0)



# Graph for specific country

#specific_country = []
#for i in vacdata:
#    if i == tkvar.get():
#        specific_country.append(i)


specific_country = [vacdata.loc[vacdata.country == 'Denmark']]
specific_country.set_value(0, 'people_fully_vaccinated', 0)
vacdata.loc[vacdata['country']=='Denmark']['people_fully_vaccinated'].astype(float).interpolate().astype(int)


result = pd.concat(specific_country)
fig = px.line(result, x='date', y='people_fully_vaccinated', color='country')

fig.update_layout(
    title={
            'text': "People fully vaccinated",
            'y':0.95,
            'x':0.5
    },
    xaxis_title="Date",
    yaxis_title="Vaccinations"
)

fig.show()


# graph for all countries
#fig = px.line(result, x='date', y='people_fully_vaccinated', color='country')

#fig.update_layout(
 #   title={
  #          'text': "People fully vaccinated",
   #         'y':0.95,
    #        'x':0.5
     #   },
    #xaxis_title="Date",
    #yaxis_title="Vaccinations"
#)
#fig.show()


#for i in countries:
 #   i = mb.menu.add_checkbutton(label=i, variable=i)

root.mainloop()