# CST8921 – Cloud Industry Trends

# Lab 1 Report – Introduction to AI and Machine Learning with Azure Machine Learning

---

## Section 1 – Cover Page

**Student Name:** Jingjing Duan
**Student ID:** 041159829
**Course:** CST8921 – Cloud Industry Trends
**Lab:** Lab 1 – Introduction to AI and Machine Learning with Azure Machine Learning

---

# Section 2 – Introduction

Azure Machine Learning (Azure ML) is a cloud-based platform provided by Microsoft for building, training, testing, and managing machine learning models. It allows developers and data professionals to create machine learning workflows using visual tools, automated systems, or programming languages such as Python.

In this lab, I used Azure Machine Learning Designer, which is a drag-and-drop visual environment inside Azure ML Studio. The Designer makes machine learning easier for beginners because users can create complete pipelines without writing code. By connecting different components together, users can perform data cleaning, model training, testing, and evaluation visually.

Machine learning and cloud AI services are becoming increasingly important in 2025 because many industries now rely on data-driven decision making. Cloud platforms such as Microsoft Azure allow organizations to process large amounts of data efficiently without maintaining expensive local infrastructure. Azure Machine Learning is useful for businesses because it supports scalable AI development, collaboration, automation, and cloud deployment.

This lab introduced the basic workflow of a machine learning regression pipeline using Azure Machine Learning Designer. The objective was to build a model that predicts automobile prices using historical automobile feature data.

---

# Section 3 – Lab Walkthrough with Screenshots

## Screenshot A1 – Workspace Deployment Completion

![alt text](/image/A1.png)

This screenshot shows the successful deployment of the Azure Machine Learning workspace in the Azure Portal. The deployment includes the workspace, storage account, key vault, and related Azure resources.

---

## Screenshot A2 – Azure ML Workspace Overview

![alt text](/image/A2.png)

This screenshot shows the Azure Machine Learning workspace overview page in the Azure Portal. The workspace was created in the Canada Central region and can launch Azure ML Studio.

---

## Screenshot B1 – AML Studio Navigation Pane

![alt text](/image/B1.png)

This screenshot shows the Azure ML Studio interface with the Authoring, Assets, and Manage sections visible. These sections organize machine learning development tools, resources, and compute infrastructure.

---

## Screenshot C1 – Compute Instance Running

![alt text](/image/C1.png)

This screenshot shows the compute instance named “lab1-compute” in the Running state. The compute instance provides the cloud virtual machine required to execute the machine learning pipeline.

---

## Screenshot D1 – Blank Designer Canvas

![alt text](/image/D1.png)

This screenshot shows the Azure ML Designer canvas with the pipeline renamed to “Automobile price prediction.” This is where the machine learning workflow is visually constructed.

---

## Screenshot E1 – Dataset Component Added

![alt text](/image/E1.png)

This screenshot shows the “Automobile price data (Raw)” dataset component added onto the Designer canvas. This dataset contains automobile information used for regression training.

---

## Screenshot E2 – Dataset Preview

![alt text](/image/E2.png)

This screenshot shows the dataset preview window. The dataset contains multiple automobile feature columns such as horsepower, engine size, and price.

---

## Screenshot F1 – Column Selection Rules

![alt text](/image/F1.png)

This screenshot shows the Select Columns in Dataset configuration. All columns were included except the “normalized-losses” column because it contained many missing values.

---

## Screenshot F2 – First Three Connected Components

![alt text](/image/F2.png)

This screenshot shows the first stage of the pipeline including the dataset, Select Columns in Dataset, and Clean Missing Data components connected together.


---

## Screenshot F3 – Split Data

![alt text](/image/F3.png)


---

## Screenshot G1 – Training Components Connected

![alt text](/image/G1.png)

This screenshot shows the Split Data, Linear Regression, and Train Model components connected correctly. The model was trained using 70% of the dataset.

---

## Screenshot H1 – Final Pipeline

![alt text](/image/H1.png)

This screenshot shows the completed machine learning pipeline with all seven components connected from data input to evaluation.

---

## Screenshot I1 – Runtime Settings

![alt text](/image/I1.png)

This screenshot shows the runtime configuration for the pipeline submission. The compute instance “lab1-compute” was selected as the execution target.

---

## Screenshot J1 – Job Detail Page

![alt text](/image/J1.png)

This screenshot shows the pipeline job execution page with completed components displayed in green. The pipeline successfully executed in Azure Machine Learning.

---

## Screenshot J2 – Score Model Output

![alt text](/image/J2.png)

This screenshot shows the output of the Score Model component. The “Scored Labels” column contains the predicted automobile prices generated by the model.

---

## Screenshot J3 – Evaluation Metrics

![alt text](/image/J3.png)

This screenshot shows the evaluation results for the regression model. Metrics such as MAE, RMSE, and R² were generated to measure model performance.

---

## Screenshot K1 – Resource Group Deleted

![alt text](/image/K1.png)

This screenshot confirms that the resource group “aml-lab-rg” was deleted successfully. This step prevents unnecessary Azure credit usage.

---

# Section 4 – Analysis and Reflection

## Reflection Question
For a real-time fraud detection system with five years of transaction data and custom preprocessing logic, I would choose the Notebooks tool from the Authoring section. Notebooks provide full programming flexibility using Python and are more suitable for advanced machine learning workflows and custom data processing.\n\nFor the compute type, I would choose Compute Clusters because they support scalable distributed computing and are better for handling large datasets and intensive machine learning training workloads.\n\nThis approach would provide better flexibility, scalability, and performance for enterprise-level fraud detection systems.

## Question 1

The regression model achieved an R² value of 0.868204. Based on the provided scale, the model performance can be classified as Good. A higher R² value means the model can better explain the relationship between automobile features and price.

---

## Question 2

The Mean Absolute Error (MAE) value was 1773.614473. This means that, on average, the predicted automobile price differed from the actual automobile price by approximately $1774.

---

## Question 3

Example row:

* Actual Price: 13495
* Predicted Price: 12800

Percentage Error Calculation:

Percentage Error = |Actual Price – Predicted Price| / Actual Price × 100%

Result: = |13495 - 12800| / 13495 × 100
        = 695 / 13495 × 100
        ≈ 5.15%

The prediction error for this automobile was approximately 5.15%, which indicates that the prediction was relatively close to the actual vehicle price.

---

## Question 4

It is important to evaluate the model using data that was not used during training because it measures how well the model performs on unseen data. If the same dataset were used for both training and testing, the model might simply memorize the data instead of learning meaningful patterns. This problem is called overfitting.

---

## Question 5

Removing entire rows with missing values helps ensure that the dataset contains complete and clean records. However, this approach reduces the total amount of data available for training. Replacing missing values with the column mean preserves more data but may introduce inaccurate or artificial values. Row removal is safer for beginner workflows, while mean replacement may be useful for larger datasets.

---

## Question 6

If the “normalized-losses” column had been kept without handling the missing values properly, the model performance would likely decrease. Missing values can cause training errors or reduce the quality of the regression model because the algorithm may not process incomplete data correctly.

---

## Question 7

Three automobile features that likely influence price are:

1. Horsepower – More powerful cars are usually more expensive.
2. Engine Size – Larger engines often indicate higher vehicle performance and price.
3. Curb Weight – Heavier vehicles are commonly associated with larger and more premium automobile models.

These features likely have strong relationships with automobile market value.

---

# Section 5 – Conclusion

In this lab, I learned how to use Azure Machine Learning Designer to create a complete machine learning regression pipeline in the cloud. The lab introduced important concepts such as data preprocessing, train/test splitting, regression model training, prediction scoring, and model evaluation.

I also learned how Azure ML integrates cloud infrastructure with machine learning workflows. The visual Designer environment made it easier to understand how machine learning pipelines operate step by step.

A similar regression pipeline could be used in many real-world business scenarios. For example, real estate companies could use regression models to predict housing prices based on features such as location, house size, and property condition. This type of predictive analytics can help businesses make more informed decisions.

---

# Section 6 – Challenges Encountered

One challenge during the lab was understanding how the pipeline components connect together correctly. It was important to connect the correct output ports between components, especially during the train/test split stage.

Another challenge was waiting for the compute instance and pipeline jobs to finish running in Azure. Sometimes cloud resources required several minutes before becoming available.

I resolved these issues by carefully reviewing the component connections and monitoring the pipeline job status inside Azure ML Studio.
