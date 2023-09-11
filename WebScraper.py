import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as AC
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import html
from lxml import etree
from lxml import html as html2
from scipy.cluster.hierarchy import dendrogram, linkage

def get_soup(url):
    response = requests.get(url)
    return BeautifulSoup(response.text, 'html.parser')

# Function to recursively extract tags and their contents
def extract_tags_contents(soup_element):
    tags_contents_list = []
    for child in soup_element.children:
        if child.name:  # Check if it's a tag
            tags_contents_list.append({'tag': child.name, 'contents': child.string})
            tags_contents_list.extend(extract_tags_contents(child))
    return tags_contents_list

def get_html(url):
    response = requests.get(url)
    html_code = response.text
    return(html_code)
#html_code = get_html('http://thdl.ntu.edu.tw/L303_SongHuiYao/RetrieveDocs.php?username=*anonymous*')

def get_xpath(html_code):
    tree = etree.fromstring(html_code)
    xpath_list = []
    for element in tree.iter():
        # Get the XPath of the element
        xpath = tree.getroottree().getpath(element)
        xpath_list.append(xpath)
    return xpath_list

def tag_split(xpath_list):
    tag_list = []
    for xpath in xpath_list:
        tags = xpath.split('/')[1:]
        tag_list.append(tags)
    return tag_list

def lastchild(df):
    parents = []
    lastchild = []
    indices = []
    list1 = sorted(df['XPath'].tolist())
    for i in range(df.shape[0]):
        if i < df.shape[0]-1:
            if list1[i] in list1[i+1]:
                parents.append(list1[i])
            else:
                lastchild.append(list1[i])
                indices.append(df[df['XPath'] == list1[i]].index.tolist()[0])
        if i == df.shape[0] -1:
            lastchild.append(list1[i])
            indices.append(df[df['XPath'] == list1[i]].index.tolist()[0])


    return lastchild, indices

def index_tag(tag_list):
    index_tag_list = []
    for tags in tag_list:
        index_tags = []
        for tag in tags:
            index_tag = str(tags.index(tag)) + tag
            index_tags.append(index_tag)
            tag_list[tag_list.index(tags)][tags.index(tag)] = ''
        index_tag_list.append(index_tags)
    return index_tag_list

def index_xpath(index_tag_list):
    indexed_xpath_list = []
    indexed_xpath = '/'
    for index_tags in index_tag_list:
        for index_tag in index_tags:
            indexed_xpath += index_tag + '/'
        indexed_xpath = indexed_xpath[:-1]
        indexed_xpath_list.append(indexed_xpath)
        indexed_xpath = '/'
    return indexed_xpath_list

def dummyinit(df):
    for value in df['Indexed XPath']:
        tags = value.split('/')[1:]
        for tag in tags:
            tag = tag.split('[')[0]
            if tag not in df.columns:
                df[tag] = None
    return df

def dummy(df, index_tag_list):
    for index_tags in index_tag_list:
        for index_tag in index_tags:
            if index_tag in list(df['Indexed XPath'])[index_tag_list.index(index_tags)]:
                df[index_tag.split('[')[0]][index_tag_list.index(index_tags)] = 1
    return df

def none2zero(x):
    if x == None:
        return 0
    else:
        return 1

def convert(df):
    for column in df.columns[3:]:
        df[column] = df[column].apply(lambda x: none2zero(x))
    return df

def length(df):
    df['Length'] = None
    df['Length'] = df.iloc[:, 3:].sum(axis = 1)
    return df

def count(df):
    df1 = df.groupby(df.columns[3:-1].tolist()).size().reset_index(name = "Count")
    return df1

def dropuseless(df2):
    for i in df2.index:
        if (('script' in df2.loc[i, 'XPath'].split('/')[-1]) | ('font' in df2.loc[i, 'XPath'].split('/')[-1]) | ('input' in df2.loc[i, 'XPath'].split('/')[-1]) | ('img' in df2.loc[i, 'XPath'].split('/')[-1]) \
            | ('option' in df2.loc[i, 'XPath'].split('/')[-1]) | ('br' in df2.loc[i, 'XPath'].split('/')[-1]) | ('hr' in df2.loc[i, 'XPath'].split('/')[-1])) & ('nobr' not in df2.loc[i, 'XPath'].split('/')[-1])\
                | ('meta' in df2.loc[i, 'XPath'].split('/')[-1]) | ('system' in df2.loc[i, 'XPath'].split('/')[-1]) | ('style' in df2.loc[i, 'XPath'].split('/')[-1]) | ('link' in df2.loc[i, 'XPath'].split('/')[-1])\
                    | ('button' in df2.loc[i, 'XPath'].split('/')[-1]) | ('title' in df2.loc[i, 'XPath'].split('/')[-1]):
            df2 = df2.drop(i)
    return df2

def countidx(df):
    df['Count_Index'] = None
    counts = [0]*len(df['Count'].unique())
    count_index = []
    for i in range(df.shape[0]):
        count = df['Count'][i]
        if count not in count_index:
            count_index.append(count)
            df['Count_Index'][i] = counts[count_index.index(count)]
            counts[count_index.index(count)] +=1
        else:
            df['Count_Index'][i] = counts[count_index.index(count)]
            counts[count_index.index(count)] +=1
    return df

def mergedf(df1, df2, columns):
    mergeddf = pd.merge(df1, df2, on = columns)
    return mergeddf

def printtags(df):
    stuff =[]
    for i in df['XPath']:
        stuff.append(i.split('/')[-1])
    stf = []
    for i in stuff:
        stf.append(i.split('[')[0])
    df = pd.DataFrame(stf, columns = ['XPath'])
    return df['XPath'].unique()

def combine(df1):
    return str(df1['Count']) + '_' + str(df1['Count_Index'])

def unique_list(series):
    unique_list = []
    for i in list(series):
        if i not in unique_list:
            unique_list.append(i)
        else:
            continue
    return unique_list

def CountIndex(df):
    l1 = unique_list(df['CountID'])
    l2 = [0]*len(unique_list(df['Count']))
    l3 = unique_list(df['Count'])
    l4 = []
    for i in l1:
        c = int(i.split('_')[0])
        l2[l3.index(c)] += 1
    for j in range(df.shape[0]):
        count = df['Count'].tolist()[j]
        l4.append(l2[l3.index(count)])
    df['CountIndex'] = l4
    return df

def clustering(arr):

    y_values = arr.reshape(-1, 1)

    # Create a linkage matrix
    Z = linkage(y_values, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.show()

    # Create an AgglomerativeClustering object and fit the data
    agglomerative = AC(n_clusters=2, linkage='ward').fit(y_values)

    # Get the cluster labels
    labels = agglomerative.labels_
    num_clusters = len(set(labels))
    clusters = []
    # Print the clusters
    for i in range(num_clusters):
        cluster_data = y_values[labels == i]
        print(f"Cluster {i+1}: {cluster_data.flatten()}")
        clusters.append(cluster_data.flatten())
    if len(clusters[0]) > len(clusters[1]):
        return clusters[0]
    else:
        return clusters[1]

def msd(cluster):
    mean = np.mean(cluster)
    sqdd = []
    for i in cluster:
        sqdd.append((i-mean)**2)
    msd = np.mean(sqdd)
    return msd

def remove_outlier(df, threshold):
    cluster = clustering(df['Count'].unique())
    while (msd(cluster) > threshold):
        cluster = clustering(cluster)
    return cluster

def filterdf(df, cluster):
    df = df[df['Count'].isin(cluster)].copy()
    return df

def patterns(df, att):
    patterns = list(df[att].unique())
    patterns.sort(reverse=True)    
    return patterns

def getelementby(df, att, patterns):
    ep = []
    elements = []
    for pattern in patterns:
        for content in df[df[att] == pattern]['Content']:
            elements.append(content)
        ep.append(elements)
        elements = []
    
    return ep

def charno(df, url):
    doc = etree.HTML(get_html(url))
    charlen = []
    chars = []
    for xpath in df['XPath']:
        element = doc.xpath(xpath)
        if len(element) > 0:
            element = html.unescape(etree.tostring(element[0]).decode())
            tree = html2.fromstring(element)
            # Extract the text content of all elements
            content = tree.xpath('//*[text()]//text()')
            content = ''.join(content)
            content = content.replace(" ", "").replace("\n", "").replace("\r", "").replace('\xa0','')
            if content == '':
                content = 'None'
                charlen.append(0)
            else:
                charlen.append(len(content))
            chars.append(content)
        else:
            charlen.append(0)
            chars.append('None')
    df['Content'] = chars
    df['Contlen'] = charlen

    return df

def mean(df):
    df1 = df.groupby('CountID')['Contlen'].mean()
    return df1

def removeone(df):
    df = df[df['Count']!= 1].reset_index()
    return df

def output(contents):
    df = pd.DataFrame(contents[0], columns = ['content'])
    return df

def main(url):
    #Get the html code
    html_code = get_html(url)
    #Get every xpath for each element in the html code
    xpath_list = get_xpath(html_code)

    #Create a dataframe with the xpaths
    df = pd.DataFrame(xpath_list, columns = ['XPath'])

    #Filter the dataframes to only include terminal elements
    terminalElements, index = lastchild(df)
    df = pd.DataFrame({'XPath': terminalElements, 'Index': index})
    df = dropuseless(df).reset_index(drop = True)
    xpath_list = df['XPath'].tolist()

    #Get the tag names for each xpath, then add the index of each tag to the tag
    tag_list = tag_split(xpath_list)
    index_tag_list = index_tag(tag_list)
    indexed_xpath = index_xpath(index_tag_list)
    df['Indexed XPath'] = indexed_xpath
    #Create dummy variables
    df = dummyinit(df)
    df = dummy(df, index_tag_list)
    df = convert(df)
    #Create a length attribute
    df = length(df)
    df1 = count(df)
    df1 = countidx(df1)
    df1['CountID'] = df1.apply(lambda x : combine(x), axis=1)
    df1 = CountIndex(df1)
    df = mergedf(df, df1,df.columns.tolist()[3:-1])
    df = filterdf(df, remove_outlier(df, 50))
    df = removeone(df)
    df = df.drop(['index'], axis = 1)
    df = charno(df, url)
    df = mergedf(df, mean(df),'CountID')
    df.columns = df.columns.str.replace('Contlen_x','Contlen').str.replace('Contlen_y','ContlenMean')
    pattern = patterns(df, 'ContlenMean')
    contents = getelementby(df,'ContlenMean', pattern)
    df1 = output(contents)
    df1.to_csv('output.csv', index = False)
main('http://thdl.ntu.edu.tw/L303_SongHuiYao/RetrieveDocs.php?username=*anonymous*')



