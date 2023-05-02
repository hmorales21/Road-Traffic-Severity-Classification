<h1>Road Traffic Severity Classification</h1>
<h2>Project 1</h2>
<hr>
<h2>Description:</h2>
<p>This data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms. 


<h2>Problem Statement:</h2> 
<p>The target feature is **Accident_severity** which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. The metric for evaluation will be **f1-score**
<hr>
<h2>Project structure</h2>
<h3>Project 1</h3>
<ul>
    <li>datasets</li>
        <p>RTA Dataset.csv - original</p>
        <p>RTA Dataset_preprocessing.csv - after EDA</p>
        <p>RTA Dataset_cleaned.csv - after preprocessing</p>
        <p>RTA Dataset_encoded.csv - after feature selection, ready to ingest into the model</p>
    <li>notebooks</li>
        <p>project_1_classification_task1_EDA_final</p>
        <p>project_1_classification_task2_Preprocessing_final</p>
        <p>project_1_classification_task2_Feature_selection.ipynb</p>
        <p>project_1_classification_task3_model.ipynb</p>
    <li>models</li>
        <p>RTA_model.joblib : Model created on feature selection notebook</p>
    <li>deployment</li>
    <p>Streamlit application</p>
</ul>
<hr>
<p>Horacio Morales González

