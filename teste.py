import streamlit as st
import math
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import mode
import altair as alt
import cryptography
from cryptography.fernet import Fernet
from PIL import Image
import base64

@st.cache(allow_output_mutation=True)
def Preenchendo_com_media(data, column, optimalK):
    feat = pd.DataFrame(data[column])
    
    feat = feat[feat.applymap(isnumber)]#Exclui valores não numéricos
    #print(feat)
    data[column] = pd.to_numeric(feat[column],errors='coerce')
    #data[column+"_isnull"] = data[column].isnull()
    #for i in range(feat.shape[0]):
        #print(type(feat.loc[i,column]))
        #print(feat.loc[i,column])
    #for i in range(data.shape[0]):
        #print(type(data.loc[i,column]))
        #print(data.loc[i,column])
        
    for i in range(optimalK):
        aux = i
        file_0 = data[data["Label"] == aux]
        #print(file_0) 
        file_0.loc[file_0[column].isnull(), column] = file_0[column].mean()
        dan = file_0.index.tolist()
        for j in dan:
                data.loc[j,column] = file_0.loc[j, column]
        #print(file_0[column].mean())    
        #print(file_0[column])
    return data

@st.cache(allow_output_mutation=True)
def Preenchendo_com_moda(data, column, optimalK):
    
    #data[column+"_isnull"] = data[column].isnull()
    for i in range(optimalK):
        aux = i
        file_0 = data[data["Label"] == aux]
        #print(file_0) 
        file_0[column].fillna(mode(file_0[column]).mode[0], inplace=True)
        dan = file_0.index.tolist()
        for j in dan:
                data.loc[j,column] = file_0.loc[j, column]    
        #print(file_0[column])
    return data

def data_coord(data, lat, lon):
    '''
    #2 Cria um DataFrame apenas com as coordenadas
    '''
    pos_df = pd.DataFrame(data, columns = [lat, lon])
    return pos_df

@st.cache(allow_output_mutation=True)
def clusterOptimalK(data, lat, lon, optimalK):
    kmeans = KMeans(n_clusters = optimalK)
    #Garante que apenas as coordenadas sejam agrupadas, 
    #evitando que outros dados influenciem nos grupos
    dataframe = pd.DataFrame(data, columns = [lat,lon]) 
    kmeans.fit_transform(dataframe)
    return kmeans

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    return f'<a href="data:file/csv;base64,{b64}" download="dados_tratados.csv">Baixar arquivo CSV</a>'
#################################################################################################################
# Funções antigas reaproveitadas

def decimal (position):
    dd = position
    if '°' in position and '\'' in position :
        x = re.split('°|\'|"', position) 
    
        deg = int(x[0])
    
        min = int(x[1])
    
        sec = float(x[2])
        
        dd = deg + (min/60) + (sec/3600)  
        dd = -dd
    else:
        dd = None
    return dd

def replacedot (dataset,columns):
    col = columns
    for x in col:
        aux = 0
        for y in dataset[x]:
            dataset.loc[aux,x] = str(dataset.loc[aux,x]).replace(",",".")
            aux+=1
    return dataset        

def removedot (dataset,columns):
    col = columns
    for x in col:
        aux = 0
        for y in dataset[x]:
            dataset.loc[aux,x] = str(dataset.loc[aux,x]).replace(".","")
            aux+=1
    return dataset        

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def Encontrando_K_Ideal(data):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,50), size = (1080,720))
    visualizer.fit(data)
    visualizer.show(outpath='teste_plot.png')
    #st.pyplot()
    return visualizer#.elbow_value_

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def frat_mult (text):
    values = re.split('/', text)
    y = 0
    for x in values:
        y = y + int(x)
    
    return y/len(values)
#################################################################################################################
#Teste de criptografia - chave simétrica - Funcões
def save_csv(data,data_file):
    data.to_csv(data_file, 'w')
    st.write("Dados originais salvos em arquivo")
    #with open('crypt/crypt_test.csv') as f:
    #    st.write(f.read())

def key_generate(file_name):
    #Generate the key
    key = Fernet.generate_key()
    st.write("Chave gerada")
    #st.write("Chave:")
    #st.write(key)
    
    #Save the key in a file
    file = open(file_name, 'wb') #wb = write bytes
    file.write(key)
    file.close()
    st.write("Chave salva em arquivo")

def key_getter(file_name):
    # Get the key from the file
    file = open(file_name, 'rb') # rb = read bytes
    key  = file.read()
    file.close()
    st.write("Chave pega do arquivo")
    return key
 
def encrypter(data_name,file_name, key):
    #  Open the file to encrypt
    with open(data_name, 'rb') as f:
        data_to_crypt= f.read()
        st.write("Dados originais pegos do arquivo")
        #st.write(data_to_crypt)

    fernet = Fernet(key)
    encrypted = fernet.encrypt(data_to_crypt)
    st.write("Dados originais criptografados")
    #st.write(encrypted)

    # Write the encrypted file
    with open(file_name, 'wb') as f:
        f.write(encrypted)
        st.write("Dados criptografados salvos em arquivo")
    
def decrypter(data_name,file_name, key):        
    #  Open the file to decrypt
    with open(data_name, 'rb') as f:
        data_to_descrypt = f.read()
        st.write("Dados criptografados pegos do arquivo")
        #st.write(data_to_descrypt)
    
    fernet = Fernet(key)
    decrypted = fernet.decrypt(data_to_descrypt)
    st.write("Dados criptografados foram descriptografados")
    #st.write(decrypted)

    # Write the decrypted file
    with open(file_name, 'wb') as f:
        f.write(decrypted)
        st.write("Dados descriptografados salvos em arquivo")

#################################################################################################################
#Sistema
st.set_page_config(
    page_title="Análise semi-supervisonada para geoprocessamento",
    page_icon=":globe_with_meridians:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title('Análise semi-supervisonada para geoprocessamento :earth_americas:')

with st.beta_expander("Instruções de uso"):
    st.header('1. Escolha seus dados')
    st.write("Na barra lateral, selecione um arquivo **CSV** que contem os dados desejados.")
    st.header('2. Defina o delimitador ')
    st.write("Insira no campo de texto qual caractere é o **delimitador** dos dados no arquvio CSV (por padrão é a vírgula).")
    st.header('3. Selecione a latitude e longitude')
    st.write("Seu arquivo será lido, e as colunas identificadas serão expostas para que sejam selecionadas aquelas que contêm as **coordenadas**.")
    st.header('4. Converta as coordenadas')
    st.write("Caso as coordenadas estejam em forma de **graus**, **minutos** e **segundos**, selecione o checkbox para convertê-las para a forma **decimal**. ")
    st.header('5. Selecione as outras colunas')
    st.write("Escolha quais colunas devem ter seus dados faltantes preenchidos pela **média** ou pela **moda** do grupo ao qual pertencer-a após o agrupamento")
    st.header('6. Defina o \'K\'')
    st.write("Caso marque este checkbox, surgirá um campo no qual deve se deve informar a quantidade desejada de grupos para dividir os dados. Caso não marque, o sistema definira a quantidade ideal de grupos por meio de algoritmos de **_machine learning_**")
    st.header('7. Executar agrupamento')
    st.write("Um **_dataframe_** com os dados será exibido no centro da tela, com as coordenadas já convertidas (caso tenha optado por tal). Vale ressaltar que na conversão, todos os dados que não se encaixem no formato serão **excluídos**. Selecione o checkbox para agrupar os dados.")

st.subheader('Data Frame:')


data = pd.DataFrame([
    [0,15,'Azul','06°26\'14.1"','36°41\'51.9"'],
    [np.nan,8,'Preto','07°21\'47.9"','36°40\'54.9"'],
    [4,np.nan,'Verde',8.1587,9.5894],
    [np.nan,19,np.nan,3.5412,12.6427]
    ],columns=['Feat0','Feat1','Feat2','Latitude', 'Longitude'])

#Entrada de dados

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo csv", type='csv')

delimiter = st.sidebar.text_input("Delimitador: (Padrão: ' , ')", value = ',')
if delimiter != ',':
    st.sidebar.warning("Verifique o estado do dataframe na exibição ao lado")

if uploaded_file is not None:
    data = pd.DataFrame(pd.read_csv(uploaded_file, delimiter = delimiter)) 
else:
    st.info("Estes dados são apenas exemplos. Selecione um arquivo para poder executar")
#Seleção de colunas

lat = st.sidebar.selectbox('Latitude:',data.columns)
lon = st.sidebar.selectbox('Longitude:',data.columns)
if lat == lon :
        st.sidebar.warning("Latitude e longitude estão iguais")
conv = st.sidebar.checkbox("Converter de DMS para DD")
mean_col = st.sidebar.multiselect("Dados a serem preenchidos com média",data.columns)
mode_col = st.sidebar.multiselect("Dados a serem preenchidos com moda",data.columns)

optimalK = 0
yellowK = True

if st.sidebar.checkbox("Definir K:"):
    optimalK = st.sidebar.number_input('Optimal K',min_value=1, value=1)
    yellowK = False

#Pré-execeução
#Conversão de DMS para DD (qualquer valor que não contenha "º" e "'" será descartado)

coord = data_coord(data, lat, lon)
if conv:
    coord = coord.astype('str').applymap(decimal)
  
data[lat] = coord[lat]
data[lon] = coord[lon]

st.dataframe(data)

#Execução mesmo
if uploaded_file is not None:
    ex = st.checkbox("Executar agrupamento")

    if ex:
        with st.spinner("Executando..."):
                coord = coord.dropna()
                coord = coord.reset_index(drop = True)

                if yellowK:
                    yellowReturn = Encontrando_K_Ideal(coord) 
                    optimalK = yellowReturn.elbow_value_
                    st.subheader('Gráfico do método do cotovelo:')
                    plot = Image.open("teste_plot.png")
                    st.image(plot, width = 830)
                    #st.image(plot, caption='Método do cotovelo', width = 830)
                    st.write('K ideal: ',optimalK)


                kmeans = clusterOptimalK(coord, lat, lon, int(optimalK)  )
                coord['Labels'] = kmeans.labels_
                
                #Preenchimento
                data = data.dropna(subset =[lat,lon])
                data = data.reset_index(drop = True)
                data['Label']  = coord['Labels']
                tool_feat = data.columns.tolist()
                if mean_col:
                    #mean_col
                    for mean in mean_col:
                        data = Preenchendo_com_media(data, mean, optimalK)
                
                if mode_col:    
                    #mode_col
                    for modes in mode_col:
                        data = Preenchendo_com_moda(data, modes, optimalK)

                #if st.button("Visualizar agrupamento"):
                #Visualização
                st.subheader('Dados tratados e preenchidos:')
                st.write(data)
                st.markdown(get_table_download_link(data), unsafe_allow_html=True)
                #st.write(coord)
                col1, col2 = st.beta_columns([1,3])
                with col1:
                    st.subheader('Coordenadas dos centróides:')
                    #Lembrar de renomear as colunas 
                    st.write(kmeans.cluster_centers_)

                #plt.figure( figsize=(15, 10))
                #scatter_plot = plt.scatter(coord[lon],coord[lat], alpha = 1, c = coord['Labels'], cmap="viridis")
                #plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s = 100, c = 'red',label = 'Centroids')
                #plt.xlabel('Longitude')
                #plt.ylabel('Latitude')
                #plt.legend()
                #st.pyplot()
                        
                #st.write(tool_feat)
                #st.write(data.columns)

                #Exibição dos grupos
                with col2:
                    #Visualização geral
                    st.subheader('Gráfico de dispersão dos dados:')
                    selection = alt.selection_multi(fields=['Label'], bind='legend')

                    all_groups = alt.Chart(data).mark_circle(size=120).encode(
                    alt.X(lon, scale = alt.Scale(zero = False)) ,
                    alt.Y(lat, scale = alt.Scale(zero = False)),
                    color=alt.condition(selection, 'Label:N', alt.value('lightgray'), legend = alt.Legend(title = 'Grupos:')),
                    tooltip = [alt.Tooltip(c, type = "nominal") for c in tool_feat
                    ]
                    ).interactive().properties(
                        width = 700,
                            height = 450
                    ).add_selection(
                    selection
                    )

                    st.altair_chart(all_groups)

                #Visualização específica
                st.subheader('Observação individual dos grupos:')    
                selected_group = st.selectbox('Selecione um grupo:', range(optimalK))

                selection = alt.selection_multi(fields=['Label'], bind='legend')

                data_select = data[data["Label"] == selected_group]

                each_group = alt.Chart(data_select).mark_circle(size=120).encode(
                alt.X(lon, scale = alt.Scale(zero = False)) ,
                alt.Y(lat, scale = alt.Scale(zero = False)),
                tooltip = [alt.Tooltip(c, type = "nominal") for c in tool_feat
                ]
                ).interactive().properties(
                    width = 700,
                    height = 450
                ).add_selection(
                selection
                )

                st.altair_chart(each_group)

                st.write(data_select.describe(include = 'all'))
                #--------------------------------------------------------------------------------------
                
                st.subheader('Mapa:')
                map = coord
                map.drop('Labels', axis = 1)
                map = map.rename(columns={lat:'latitude', lon:"longitude"})
                st.map(map)

                #df = pd.DataFrame(
                #np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
                #columns=[lat, lon])

                #st.pydeck_chart(pdk.Deck(
                    #map_style='mapbox://styles/mapbox/light-v9',
                    #initial_view_state=pdk.ViewState(
                        #latitude=37.76,
                        #longitude=-122.4,
                        #zoom=11,
                        #pitch=50,
                    #),
                    #layers=[
                        #pdk.Layer(
                            #'HexagonLayer',
                            #data=df,
                            #get_position='[Longitude, Latitude]',
                            #radius=200,
                            #elevation_scale=4,
                            #elevation_range=[0, 1000],
                            #pickable=True,
                            #extruded=True,
                        #),
                        #pdk.Layer(
                            #'ScatterplotLayer',
                            #data=df,
                            #get_position='[Longitude, Latitude]',
                            #get_color='[200, 30, 0, 160]',
                            #get_radius=200,
                        #),
                    #],
                #))
            ##########################################################################
                #Teste de criptografia - chave simétrica - Execução
                
                #key_file = 'crypt/key.key'
                #data_file = 'crypt/crypt_test.csv'
                #crypt_file = 'crypt/test.csv.encrypted'
                #decrypt_file = 'crypt/secretdf.csv.decrypted'

                #save_csv(data,data_file)
                #key_generate(key_file)
                #encrypter(data_file, crypt_file, key_getter(key_file))
                #decrypter(crypt_file, decrypt_file, key_getter(key_file))
        
#chart_data = pd.DataFrame(
#    np.random.randn(20, 3),
#    columns=['a', 'b', 'c'])

#st.line_chart(chart_data)

