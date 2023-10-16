import base64
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import mlflow
from urllib.parse import urlparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



data_url = "http://lib.stat.cmu.edu/datasets/boston" 



# setting up the page streamlit

st.set_page_config(
    page_title="Linear Regression App ", layout="wide", page_icon="./images/linear-regression.png"
)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

image_nyu = Image.open('images/nyu.png')
st.image(image_nyu, width=100)

st.title("Linear Regression Lab 🧪")

# navigation dropdown

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction 🔥','Visualization 📊','Prediction 🌠','Deployment 🚀'])
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Wine Quality 🍷","Real Estate 🏰","student score 💯","Adult Income 💵"])
if "Wine Quality 🍷" in select_dataset:
    df = pd.read_csv("wine_quality_red.csv")
elif "Real Estate 🏰" in select_dataset: 
    df = pd.read_csv("real_estate.csv")
elif "Adult Income 💵" in select_dataset:
    df = pd.read_csv("adult_income.csv")
else:
    df = pd.read_csv("Student_Performance.csv")

df = df.dropna()
list_variables = df.columns

# page 1 
if app_mode == 'Introduction 🔥':
    image_header = Image.open('./images/Linear-Regression1.webp')
    st.image(image_header, width=600)
    file_ = open("images/lr2.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="lr gif">',
        unsafe_allow_html=True,
    )

    st.markdown("### 00 - Show  Dataset")

    # Wine Quality dataset
    if select_dataset == "Wine Quality 🍷":
        
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid blue;
                    text-align: center;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(3)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(4)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(5)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(6)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(7)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(8)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(9)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(10)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
            </style>
            """,unsafe_allow_html=True
        )

        col1, col2, col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
        col1.markdown(" **fixed acidity** ")
        col1.markdown("most acids involved with wine or fixed or nonvolatile (do not evaporate readily)")
        col2.markdown(" **volatile acidity** ")
        col2.markdown("the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste")
        col3.markdown(" **citric acid** ")
        col3.markdown("found in small quantities, citric acid can add 'freshness' and flavor to wines")
        col4.markdown(" **residual sugar** ")
        col4.markdown("the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter")
        col5.markdown(" **chlorides** ")
        col5.markdown("the amount of salt in the wine")
        col6.markdown(" **free sulfur dioxide** ")
        col6.markdown("the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents ")
        col7.markdown(" **total sulfur dioxide** ")
        col7.markdown("amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 ")
        col8.markdown(" **density** ")
        col8.markdown("the density of water is close to that of water depending on the percent alcohol and sugar content")
        col9.markdown(" **pH** ")
        col9.markdown("describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the ")
        col10.markdown(" **sulphates** ")
        col10.markdown("a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobia")
    
    #real estate
    elif select_dataset == "Real Estate 🏰":
        
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid blue;
                    text-align: center;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(3)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(4)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(5)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(6)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(7)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(8)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(9)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(10)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(11)
                    {
                        border:1px solid blue;
                        text-align: center;
                    } 
                div[data-testid="column"]:nth-of-type(12)
                    {
                        border:1px solid blue;
                        text-align: center;
                    } 

            """,unsafe_allow_html=True
        )

        col1, col2, col3,col4,col5,col6,col7,col8,col9,col10,col11,col12 = st.columns(12)
        col1.markdown(" **CRIM** ")
        col1.markdown("per capita crime rate by town")
        col2.markdown(" **ZN** ")
        col2.markdown("proportion of presidential land zoned for lots over 25,000 sq.ft.")
        col3.markdown(" **CHAS** ")
        col3.markdown("Charles River dummy variable (= 1 if tract bounds river; 0 otherwise")
        col4.markdown(" **NOX** ")
        col4.markdown("nitric oxide concentration (parts per 10 million)")
        col5.markdown(" **INDUS** ")
        col5.markdown("proportion of non-retail business acres per town") 
        col6.markdown(" **AGE** ")
        col6.markdown("proportion of owner-occupied units built prior to 1940")
        col7.markdown(" **DIS** ")
        col7.markdown("weighted distances to five Boston employment centres")
        col8.markdown(" **RAD** ")
        col8.markdown("index of accessibility to radial highways")
        col9.markdown(" **TAX** ")
        col9.markdown("full-value property-tax rate per $10,000")
        col10.markdown(" **PTRATIO** ")
        col10.markdown("pupil-teacher ratio by town")                        
        col11.markdown(" **B** ")
        col11.markdown("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
        col12.markdown(" **LSTAT** ")
        col12.markdown("percentage lower status of the population") 

        # student score 💯
    elif select_dataset == "student score 💯":
        
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid blue;
                    text-align: center;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(3)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(4)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(5)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(6)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                
            </style>
            """,unsafe_allow_html=True
        )

        col1, col2, col3,col4,col5,col6,= st.columns(6)
        col1.markdown(" **Hours Studied** ")
        col1.markdown("The total number of hours spent studying by each student")
        col2.markdown(" **Previous Scores** ")
        col2.markdown("The scores obtained by students in previous tests")
        col3.markdown(" **Extracurricular Activities** ")
        col3.markdown("Whether the student participates in extracurricular activities (Yes or No)")
        col4.markdown(" **Sleep Hours** ")
        col4.markdown("The average number of hours of sleep the student had per day")
        col5.markdown(" **Sample Question Papers Practiced** ")
        col5.markdown("The number of sample question papers the student practiced")
        col6.markdown(" **Performance Index** ")
        col6.markdown("A measure of the overall performance of each student")
        


    # Adult Income dataset
    elif select_dataset == "Adult Income 💵":
        
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid blue;
                    text-align: center;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(3)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(4)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(5)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(6)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(7)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(8)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(9)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(10)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(11)
                    {
                        border:1px solid blue;
                        text-align: center;
                    } 
                div[data-testid="column"]:nth-of-type(12)
                    {
                        border:1px solid blue;
                        text-align: center;
                    } 
            </style>
            """,unsafe_allow_html=True
        )

        col1, col2, col3,col4,col5,col6,col7,col8,col9,col10,col11 = st.columns(11)
        col2.markdown(" **Workclass** ")
        col2.markdown("represents the employment status of an individua")
        col3.markdown(" **fnlwgt** ")
        col3.markdown("Final Weight - this is the number of people the census believes the entry represents")
        col4.markdown(" **Education Num** ")
        col4.markdown("the highest level of education achieved in numerical form")
        col5.markdown(" **Occupation** ")
        col5.markdown("The general type of occupation of an individual")
        col6.markdown(" **Relationship** ")
        col6.markdown("represents what this individual is relative to others")
        col7.markdown(" **Race** ")
        col7.markdown("Descriptions of an individual’s race")
        col8.markdown(" **Capital Gain** ")
        col8.markdown(" capital gains for an individual")
        col9.markdown(" **Capital Loss** ")
        col9.markdown("capital loss for an individual") 
        col10.markdown(" **Hours per Week** ")
        col10.markdown("the hours an individual has reported to work per week") 
        col11.markdown(" **Native Country** ")
        col11.markdown("country of origin for an individual")
        col1.markdown(" **The Label** ")
        col1.markdown("whether or not an individual makes more than $50,000 annually") 
        df = df.drop(['workclass','education','occupation','race'],axis=1)



    
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        if st.button("Show Code"):
            code = '''df.head(num)'''
            st.code(code, language='python')
        st.dataframe(df.head(num))
    else:
        if st.button("Show Code"):
            code = '''df.tail(num)'''
            st.code(code, language='python')
        st.dataframe(df.tail(num))
    
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.markdown("### 01 - Description")
    if st.button("Show Describe Code"):
        code = '''df.describe()'''
        st.code(code, language='python')
    st.dataframe(df.describe())


    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    #df = df.dropna()
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    if st.button("Show the Code"):
        code = ''' dfnull = df.isnull().sum()/len(df)*100 
totalmiss = dfnull.sum().round(2)
        '''
        st.code(code, language='python')
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good as we have less then 30 percent of missing values.")
        st.image(
            "https://media.giphy.com/media/3ohzdIuqJoo8QdKlnW/giphy.gif",
            width=400,
        )
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    
    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    if st.button("Show the ratio Code"):
        code = ''' nonmissing = (df.notnull().sum().round(2))
completeness= round(sum(nonmissing)/len(df),2)
        '''
        st.code(code, language='python')
    st.write(nonmissing)
    
    if completeness >= 0.80:
        st.success("Looks good as we have completeness ratio greater than 0.85.")
        st.image(
            "https://media.giphy.com/media/3ohzdIuqJoo8QdKlnW/giphy.gif",
            width=400,
        )
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")


    st.markdown("### 04 - Complete Report")

    if st.button("Generate Report"):
        pr = df.profile_report()
        export=pr.to_html()
        if st.download_button(label="Download Full Report", data=export,file_name='report.html'):
            print("Downloaded")
        else:
            st_profile_report(pr)   
    

    #pr = df.profile_report()
    
    
    


# page 2
if app_mode == 'Visualization 📊':
    st.markdown("## Visualization")
    if select_dataset == "Wine Quality 🍷":
        symbols = st.multiselect("Select two variables",list_variables,["sulphates","volatile acidity"] )
   
    elif select_dataset == "Real Estate 🏰":
        symbols = st.multiselect("Select two variables",list_variables,["CRIM","NOX"] )

    elif select_dataset == "student score 💯":
        symbols = st.multiselect("Select two variables",list_variables,["Hours Studied","Performance Index"] )

    elif select_dataset == "Adult Income 💵":
        symbols = st.multiselect("Select two variables",list_variables, ["occupation","education"] )

    width1 = st.sidebar.slider("plot width", 1, 25, 10)
    #symbols = st.multiselect("", list_variables, list_variables[:5])
    tab1, tab2= st.tabs(["Line Chart","📈 Correlation"])    

    #tab1
    if tab1.button("Show Line Chart Code"):
        code = '''st.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)'''
        tab1.code(code, language='python')
    
    tab1.subheader("Line Chart")
    tab1.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)
    tab1.write(" ")
    tab1.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)

    #tab2
    if tab2.button("Show Correlation Code"):
        code = '''sns.heatmap(df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)'''
        tab2.code(code, language='python')
    tab2.subheader("Correlation Tab 📉")
    fig,ax = plt.subplots(figsize=(width1, width1))
    sns.heatmap(df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)
    tab2.write(fig)


    st.write(" ")
    st.write(" ")
    st.markdown("### Pairplot")
    
    if st.button("Show pairplot Code"):
        code = '''sns.pairplot(df2)'''
        st.code(code, language='python')
    
    df2 = df[[list_variables[0],list_variables[1],list_variables[2],list_variables[3],list_variables[4]]]
    fig3 = sns.pairplot(df2)
    st.pyplot(fig3)



# page 3
if app_mode == 'Prediction 🌠':
    st.markdown("## Prediction")

    # converting data
    if select_dataset == "student score 💯":
        # Use apply with a lambda function to map values
        df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else None))

    elif select_dataset == "Adult Income 💵":
        df = pd.get_dummies(df, columns=['education'], drop_first=True)
        df = df.drop(['workclass','occupation','education.num','relationship','race','native.country'],axis=1)
        columns_to_dummy = ['marital.status', 'sex']
        df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)
        std = StandardScaler()
        mms = MinMaxScaler()
        columns_to_scaler = ['capital.gain', 'capital.loss', 'hours.per.week']
        df[columns_to_scaler] = std.fit_transform(df[columns_to_scaler]) 
        income_map = {'<=50K': 1, '>50K': 0}
        df['income'] = df['income'].map(income_map)






    #choose the dependent variable
    target_choice =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)
    st.experimental_set_query_params(saved_target=target_choice)
    
    #dropping the target column
    new_df= df.drop(labels=target_choice, axis=1)  #axis=1 means we drop data by columns
    #imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imp_mean.fit()

    #other column list
    list_var = new_df.columns
    feature_choice = st.multiselect("Select Explanatory Variables", list_var)
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    
    def predict(target_choice,train_size, new_df,feature_choice):
        #independent variables / explanatory variables
        #choosing column for target
        new_df2 = new_df[feature_choice]
        x =  new_df2
        y = df[target_choice]
        col1,col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(x.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))
        #X = df[feature_choice].copy()
        #y = df['target'].copy()
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size)
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)
        lm = LinearRegression()
        lm.fit(X_train,y_train)
        predictions = lm.predict(X_test)
        return lm,y_test,predictions

    # Mlflow tracking
    track_with_mlflow = st.checkbox("Track with mlflow? 🛤️")

    # Model training
    start_training = st.button("Start training")
    if not start_training:
        st.stop()

    if track_with_mlflow:
        #mlflow.set_tracking_uri("./model_metrics")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        experiment_name = select_dataset
        st.write(experiment_name)
        mlflow.start_run()
        try:
            # creating a new experiment
            exp_id = mlflow.create_experiment(name=experiment_name)
        except Exception as e:
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        #mlflow.set_experiment(select_dataset)
        mlflow.log_param('model', LinearRegression)
        mlflow.log_param('features', feature_choice)
    
    lm,y_test,predictions = predict(target_choice,train_size,new_df,feature_choice)
    
    # Model evaluation
    #preds_train = model.predict(X_train)
    #preds_test = model.predict(X_test)
    #preds_test = lm.predict(X_test)
    # if problem_type=="classification":
    #     st.subheader('🎯 Results')
    #     metric_name = "f1_score"
    #     metric_train = f1_score(y_train, preds_train, average='micro')
    #     metric_test = f1_score(y_test, preds_test, average='micro')
    # else:
    #     st.subheader('🎯 Results')
    mae = np.round(mt.mean_absolute_error(y_test, predictions ),2)
    mse = np.round(mt.mean_squared_error(y_test, predictions),2)
    r2 = np.round(mt.r2_score(y_test, predictions),2)
    #metric_name = "r2_score"
    #metric_test = r2_score(y_test, preds_test)
    #st.write(metric_name+"_train", round(metric_train, 3))
    #st.write(metric_name+"_test", round(metric_test, 3))
    if track_with_mlflow:
        mlflow.sklearn.log_model(lm, "top_model_v1")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.end_run()
    if mlflow.active_run():
        mlflow.end_run()

    # Save the model to a PKL file
    with open('model.pkl', 'wb') as file:
        pickle.dump(lm, file)

    # model_code = st.checkbox("See the model code? 👀")
    # if model_code:
    #     code = '''X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)'''
    #     code1 = '''lm = LinearRegression()'''
    #     code2 = '''lm.fit(X_train,y_train)'''
    #     code3 = '''predictions = lm.predict(X_test)'''
    #     st.code(code, language='python')
    #     st.code(code1, language='python')
    #     st.code(code2, language='python')
    #     st.code(code3, language='python')
    
    st.subheader('🎯 Results')

    st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    st.write("2) The Mean Absolute Error of model is:", mae)
    st.write("3) MSE: ", mse)
    st.write("4) The R-Square score of the model is " , r2)

    def download_file():
        file_path = 'model.pkl'  # Replace with the actual path to your model.pkl file
        with open(file_path, 'rb') as file:
            contents = file.read()
        b64 = base64.b64encode(contents).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="model.pkl">Download model.pkl file</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.title("Download Model Example")
        st.write("Click the button below to download the model.pkl file.")
        if st.button("Download"):
            download_file()
       # return X_train, X_test, y_train, y_test, predictions,x,y, mae,mse, r2
    

if app_mode == 'Deployment 🚀':
    

    id = st.text_input('ID Model', '00ffae4993044a5d9cb369a46dbc1e01')
        # Print emissions
    #logged_model = f'runs:/{id}/top_model_v1'
    logged_model = f'./mlruns/0/{id}/artifacts/top_model_v1'
    
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    app_state = st.experimental_get_query_params()  
    #saved_target
    if "saved_target" in app_state:
        target_choice = app_state["saved_target"][0]
    else:
        st.write("target not saved")
    
    deploy_df= df.drop(labels=target_choice, axis=1) 
    list_var = deploy_df.columns
    st.write(target_choice)
    
    number1 = st.number_input(deploy_df.columns[0],0.7)
    number2 = st.number_input(deploy_df.columns[1],0.04)
    number3 = st.number_input(deploy_df.columns[2],1.1)
    number4 = st.number_input(deploy_df.columns[3],0.05)
    number5 = st.number_input(deploy_df.columns[4],25)
    number6 = st.number_input(deploy_df.columns[5],20)
    number7 = st.number_input(deploy_df.columns[6],0.98)
    number8 = st.number_input(deploy_df.columns[7],1.9)
    number9 = st.number_input(deploy_df.columns[8],0.4)
    number10 = st.number_input(deploy_df.columns[9],9.4)
    number11 = st.number_input(deploy_df.columns[10],5)

    data_new = pd.DataFrame({deploy_df.columns[0]:[number1], deploy_df.columns[1]:[number2], deploy_df.columns[2]:[number3],
         deploy_df.columns[3]:[number4], deploy_df.columns[4]:[number5], deploy_df.columns[5]:[number6], deploy_df.columns[6]:[number7],
         deploy_df.columns[7]:[number8], deploy_df.columns[8]:[number9],deploy_df.columns[9]:[number10],deploy_df.columns[10]:[number11]})
    # Predict on a Pandas DataFrame.
    #import pandas as pd
    st.write("Prediction :",loaded_model.predict(data_new)[0])






if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### 👨🏼‍💻 **App Contributors:** ")
st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

st.markdown(f"####  Link to Project Website [here]({'https://github.com/NYU-DS-4-Everyone/Linear-Regression-App'}) 🚀 ")
st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        "👨🏼‍💻 Made by ",
        link("https://github.com/NYU-DS-4-Everyone", "NYU - Professor Gaëtan Brison"),
        "🚀"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()