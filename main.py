import pandas as pd
import numpy as np
from tkinter import *
from DataClean import *
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import linear_model
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


# Instantiate root window
#def run():
    #print (mb.get())

# Read csv files no paths pwease! Just keep data in seperate data folder
vacdata = pd.read_csv('data/country_vaccinations.csv')
popdata = pd.read_csv('data/population_by_country_2020.csv')



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

# Merge datasets
mergedata = pd.merge(vacdata, popdata_new)
mergedata_san = mergedata.dropna()
mergedata_san['vaccinated_percent'] = mergedata_san['people_fully_vaccinated'].div(mergedata_san['population'])
mergedata_san = mergedata_san.round(5)
mergedata_san = mergedata_san.drop(['population', 'people_fully_vaccinated'], axis=1)
print(mergedata_san.head(100))

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
#
# Make the value change
def change_dropdown(*args):
    print(tkvar.get())

# Link function to change dropdown
#tkvar.trace('w', change_dropdown())

predictbutton = Button(mainframe, text="Predict!").grid(row=6, column=0)



# Graph for specific country

do_it_for_all_countries = True

def interpolate_country(df, country):

    firs = df.loc[df['country'] == country, 'people_fully_vaccinated'].index[0]
    col = df.columns.get_loc('people_fully_vaccinated')
    df.iloc[firs, col] = 0
    specific_col = 'people_fully_vaccinated'
    return df.loc[vacdata['country'] == country, specific_col].interpolate(limit_direction='both', limit=df.shape[0])


# This could be better
if do_it_for_all_countries:
    for country in vacdata['country'].unique():
        vacdata.loc[vacdata['country'] == country, 'people_fully_vaccinated'] = interpolate_country(vacdata, country)
else:
    vacdata.loc[vacdata['country'] == 'Denmark', 'people_fully_vaccinated'] = interpolate_country(vacdata, 'Denmark')


fig = px.line(mergedata_san, x='date', y='vaccinated_percent', color='country')

fig.update_layout(
    title={
            'text': "Vaccinated percent",
            'y':0.95,
            'x':0.5
    },
    xaxis_title="Date",
    yaxis_title="Vaccinations percent"
)

fig.show()




"""
#creating the model
model=LinearRegression()
#print(type(vacdata['date']))
X = mergedata.san[['people_fully_vaccinated']]
y = mergedata.san['date']
model.fit(X, y)
print(model.coef_)
print(model.intercept_)
# 1 year prediction
pr = model.predict(X)
fig, ax = plt.subplots(figsize=(15, 5))
plt.title('The Best Fit Line: ')
plt.scatter(X=mergedata.san['people_fully_vaccinated'], y=mergedata.san['date'])
plt.plot(X, pr)
predictsomething = model.predict([[100000]])
print(predictsomething)
"""

root.mainloop()