# CST8921 – Cloud Industry Trends
## Lab 1: Introduction to AI and Machine Learning with Azure Machine Learning

---

| Field | Details |
|---|---|
| **Course** | CST8921 – Cloud Industry Trends |
| **Lab Number** | Lab 1 |
| **Topic** | Artificial Intelligence & Machine Learning on Azure |
| **Estimated Duration** | 2–3 Hours |
| **Difficulty** | Beginner |
| **Primary Reference** | [Microsoft Learn – Train a No-Code Regression Model in Designer](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-train-score?view=azureml-api-1) |
| **Deliverable** | Lab Report with Screenshots submitted via Brightspace Assignments |


---

## Table of Contents

1. [Overview](#1-overview)
2. [Learning Objectives](#2-learning-objectives)
3. [Prerequisites](#3-prerequisites)
4. [Background: Key Concepts](#4-background-key-concepts)
5. [Part A – Create an Azure Machine Learning Workspace](#part-a--create-an-azure-machine-learning-workspace)
6. [Part B – Explore Azure Machine Learning Studio](#part-b--explore-azure-machine-learning-studio)
7. [Part C – Create a Compute Instance](#part-c--create-a-compute-instance)
8. [Part D – Create a New Pipeline in Designer](#part-d--create-a-new-pipeline-in-designer)
9. [Part E – Import and Visualize Data](#part-e--import-and-visualize-data)
10. [Part F – Prepare Data (Clean and Transform)](#part-f--prepare-data-clean-and-transform)
11. [Part G – Train the Machine Learning Model](#part-g--train-the-machine-learning-model)
12. [Part H – Score and Evaluate the Model](#part-h--score-and-evaluate-the-model)
13. [Part I – Submit the Pipeline Job](#part-i--submit-the-pipeline-job)
14. [Part J – Review Job Results](#part-j--review-job-results)
15. [Part K – Clean Up Azure Resources](#part-k--clean-up-azure-resources)
16. [Lab Report Requirements](#lab-report-requirements)
17. [Grading Rubric](#grading-rubric)
18. [Helpful References](#helpful-references)

---

## 1. Overview

In this lab you will gain hands-on experience with **Azure Machine Learning (AML)** — Microsoft's fully managed cloud platform for building, training, deploying, and monitoring machine learning models at scale.

This is a **beginner-level** lab. No prior machine learning or advanced mathematics experience is required. You will use the **Azure Machine Learning Designer**, a visual drag-and-drop canvas, to manually construct a complete end-to-end training pipeline — step by step, component by component — that predicts automobile prices using linear regression.

Unlike loading a pre-built sample, you will **build the pipeline from scratch** following the current Microsoft tutorial approach. This means you will understand the purpose of every component before connecting it, giving you a deeper and more transferable understanding of how ML pipelines work.

> **Professor's Note:** In professional settings, data scientists rarely use visual tools for production models — they use Python SDKs and MLOps pipelines. However, the Designer is an outstanding pedagogical tool: it makes the data flow and component dependencies *visible*, which is the best way to learn the conceptual architecture before writing code. Think of this lab as learning to read a blueprint before picking up a wrench.

---

## 2. Learning Objectives

Upon completing this lab, students will be able to:

- Provision and configure an Azure Machine Learning workspace via the Azure Portal
- Navigate all three sections of AML Studio: Authoring, Assets, and Manage
- Create and start a compute instance to execute pipeline workloads
- Build a regression training pipeline in AML Designer using classic prebuilt components
- Apply data preparation steps: column selection, missing value handling, and train/test splitting
- Configure a Linear Regression model and connect it to a Train Model component
- Score and evaluate model performance using Score Model and Evaluate Model components
- Interpret key regression metrics: MAE, RMSE, RSE, and R² (Coefficient of Determination)
- Apply responsible cloud cost management by stopping and deleting all resources after use

---

## 3. Prerequisites

Before beginning this lab, ensure you have:

- An active **Azure for Students** subscription (verify at [portal.azure.com](https://portal.azure.com))
- A modern web browser (Microsoft Edge or Google Chrome recommended)
- Reliable internet connectivity
- Familiarity with the Azure Portal (creating resources, navigating menus)
- Foundational understanding of supervised machine learning concepts from course lectures

> **No software installation is required.** All activities are performed entirely in the browser.

---

## 4. Background: Key Concepts

Review these concepts before starting. They explain the *why* behind each lab step.

### 4.1 Azure Machine Learning Workspace

The workspace is the **top-level Azure resource** for all ML activities. It provides a centralised location to manage every artifact you create: datasets, models, pipelines, compute resources, and experiment runs. Every action in this lab happens within a workspace.

### 4.2 Azure Machine Learning Designer

The Designer is a **visual, no-code environment** within AML Studio for building ML pipelines. You construct pipelines by dragging components onto a canvas and connecting their input/output ports. The Designer currently supports two component types:

| Component Type | Description | Compatibility |
|---|---|---|
| **Classic prebuilt (v1)** | Pre-built components for common data processing and ML tasks | Used in this lab; supported but not receiving new additions |
| **Custom components (v2)** | Wrap your own Python code as reusable pipeline steps | Recommended for new projects; compatible with SDK v2 and CLI v2 |

> These two types are **not compatible** with each other in the same pipeline.

### 4.3 Supervised Learning and Regression

**Supervised learning** trains a model on labelled examples — data where the correct answer (label) is already known. **Regression** is a type of supervised learning where the label is a continuous number. In this lab, the label is `price` — the actual selling price of a car. The model learns the relationship between a car's features (horsepower, engine size, fuel type, etc.) and its price.

### 4.4 The ML Pipeline Architecture

Your pipeline will follow the universal ML workflow:

```
Raw Data → Select Relevant Columns → Clean Missing Values →
Split (Train 70% / Test 30%) → Train Model → Score Model → Evaluate Model
```

Each of these steps is a **component** in the Designer canvas. Understanding why each component exists is as important as knowing how to add it.

### 4.5 What Is a Compute Instance?

A compute instance is a **fully managed cloud VM** that executes your pipeline steps. You select a VM size based on your workload. You are billed per minute while it is running — which is why stopping or deleting it after use is a professional and financial responsibility.

---

## Part A – Create an Azure Machine Learning Workspace

### Step A.1 – Sign In to the Azure Portal

1. Open your browser and navigate to [https://portal.azure.com](https://portal.azure.com).
2. Sign in using the Microsoft account associated with your **Azure for Students** subscription.
3. Confirm your subscription is active by clicking your account name (top right) → **Switch directory** and verifying your student subscription is listed.

---

### Step A.2 – Create the Azure Machine Learning Resource

1. In the top search bar, type **Azure Machine Learning** and select it from the list of services.
2. Click **+ Create** to open the workspace creation wizard.
3. On the **Basics** tab, enter the following settings **exactly as shown**:

| Setting | Value |
|---|---|
| Subscription | `Azure for Students` |
| Resource Group | `aml-lab-rg` → click **Create new** and type this name |
| Workspace Name | `aml-lab-workspace` |
| Region | `Canada Central` |
| Storage Account | *(Accept the auto-generated name)* |
| Key Vault | *(Accept the auto-generated name)* |
| Application Insights | *(Accept the auto-generated name)* |
| Container Registry | `None` *(default — created automatically on first model deployment)* |

> **Why Canada Central?** Selecting the Canada Central Azure region keeps your data within Canadian borders, satisfies Canadian data residency requirements, and minimises network latency from your Algonquin College campus connection. This is a professional best practice for Canadian cloud deployments.

4. Leave the **Networking**, **Encryption**, and **Tags** tabs at their defaults.
5. Click **Review + Create**. After validation passes (green checkmark), click **Create**.
6. Deployment typically takes **3–5 minutes**. Wait for the "Your deployment is complete" notification.

>  **Screenshot Checkpoint A1:** Capture the "Your deployment is complete" page showing all resources deployed successfully, including the workspace name and region.

---

### Step A.3 – Launch Azure Machine Learning Studio

1. On the deployment completion page, click **Go to resource**.
2. On the Azure Machine Learning workspace Overview page, click **Launch studio**.
3. AML Studio opens at [https://ml.azure.com](https://ml.azure.com) in a new browser tab.
4. If prompted to select a workspace, choose **aml-lab-workspace** and click **Use this workspace**.

> **Screenshot Checkpoint A2:** Capture the AML workspace Overview page in the Azure Portal showing your workspace name, region (Canada Central), and the **Launch studio** button.

---

## Part B – Explore Azure Machine Learning Studio

Before building anything, spend a few minutes systematically exploring the Studio interface. This orientation will serve you throughout the entire course.

### Step B.1 – Explore the Left Navigation Pane

The navigation pane on the left side is divided into three sections. Expand each one and take note of its contents.

---

#### Authoring Section
Contains tools for **creating** ML models and experiments.

| Tool | Description |
|---|---|
| **Notebooks** | Managed Jupyter environment for Python/R code. Ideal for custom data exploration, feature engineering, and model development. |
| **Automated ML** | Automatically trains and evaluates many model types on your dataset and selects the best performer. No algorithm selection required. |
| **Designer** | Visual drag-and-drop pipeline builder using prebuilt or custom components. This is the tool used in today's lab. |

> **Professor's Note:** These three tools represent a spectrum from automation to control. AutoML does the most for you but gives you the least insight. Notebooks give you full control but require strong coding skills. Designer sits in the middle — it exposes the pipeline architecture without requiring code, making it ideal for learning.

---

#### Assets Section
Tracks all **objects created or consumed** during ML workflows. Assets are versioned so experiments are reproducible.

| Asset | Description |
|---|---|
| **Data** | Registered datasets (files, tables, cloud storage URIs) reusable across experiments. |
| **Jobs** | A record of every pipeline or script run — status, logs, metrics, outputs, and duration. |
| **Models** | Serialised trained model files that can be deployed or further evaluated. |
| **Pipelines** | Saved or published pipeline definitions. |
| **Environments** | Docker-based environment snapshots specifying Python packages and runtime versions. |
| **Components** | Reusable, self-contained pipeline steps (especially useful in the custom v2 model). |

---

#### Manage Section
Contains **infrastructure resources** needed to run ML workloads.

| Resource | Description |
|---|---|
| **Compute Instances** | Single-node managed VMs for interactive development and running pipelines. *Used in this lab.* |
| **Compute Clusters** | Auto-scaling multi-node clusters for large-scale training jobs. |
| **Inference Clusters** | Kubernetes clusters for deploying models as scalable REST endpoints. |
| **Attached Compute** | External compute resources (e.g., Azure Databricks) linked to this workspace. |

> **Screenshot Checkpoint B1:** Capture the AML Studio home page with the left navigation pane fully visible, showing the Authoring, Assets, and Manage sections.

### Step B.2 – Reflection Question (Answer in Lab Report)

> *If your team needed to train a real-time fraud detection model on 5 years of transaction data with custom preprocessing logic, which Authoring tool would you choose, and why? Which compute type from the Manage section would be most appropriate?*

---

## Part C – Create a Compute Instance

A compute instance must be running before you can submit a pipeline job.

### Step C.1 – Navigate to Compute

1. In the left navigation pane, under **Manage**, click **Compute**.
2. Select the **Compute instances** tab.

### Step C.2 – Create a New Compute Instance

1. Click **+ New**.
2. In the **Create compute instance** panel, configure the following:

| Setting | Value |
|---|---|
| Compute name | `lab1-compute` *(lowercase alphanumeric and hyphens only; must be unique in region)* |
| Virtual machine type | `CPU` |
| Virtual machine size | `Standard_DS11_v2` *(select from the Recommended list)* |

> **Why Standard_DS11_v2?** This general-purpose VM has 2 vCPUs and 14 GB RAM — well-suited for running a small regression pipeline. It provides a good balance of performance and Azure for Students credit consumption for this beginner lab.

> **Note from Microsoft Docs:** Attached compute is **not supported** in Designer — you must use compute instances or clusters. Do not attempt to use an externally attached compute target.

3. Leave all remaining settings at their defaults.
4. Click **Create**.

### Step C.3 – Wait for Running Status

- The instance transitions through: **Creating → Starting → Running**
- This typically takes **3–5 minutes**
- A green dot and the label **Running** confirms it is ready

> **Do not proceed until the instance shows Running status.** Submitting a pipeline to a non-running instance will cause the job to fail.

> **Screenshot Checkpoint C1:** Capture the Compute Instances list clearly showing `lab1-compute` with **Running** status, the VM size, and the creation date/time.

---

## Part D – Create a New Pipeline in Designer

You will now build a regression pipeline from scratch using the classic prebuilt components. Follow each step carefully — the connections between components define the data flow.

### Step D.1 – Open the Designer

1. In the left navigation pane, under **Authoring**, click **Designer**.
2. You will see a **New pipeline** section.

### Step D.2 – Create a New Classic Prebuilt Pipeline

Based on the current Microsoft documentation, the Designer now shows two pipeline modes:

1. Under **Classic prebuilt**, click **Create a new pipeline using classic prebuilt components**.

>  **Important:** Do **not** select "Custom component pipeline" — that uses the v2 component model, which is incompatible with the prebuilt components used in this lab.

### Step D.3 – Rename the Pipeline

1. At the top of the canvas, you will see an auto-generated draft name (e.g., *Pipeline-Created-on-date*).
2. Click the **pencil icon ✏️** next to the draft name.
3. Rename the pipeline to: **`Automobile price prediction`**

> The name does not need to be globally unique within your workspace.

>  **Screenshot Checkpoint D1:** Capture the blank Designer canvas showing the renamed pipeline title **"Automobile price prediction"** at the top.

---

## Part E – Import and Visualize Data

### Step E.1 – Add the Dataset to the Canvas

The Designer provides several built-in sample datasets. You will use the **Automobile price data (Raw)** dataset.

1. In the **component palette** on the left side of the canvas, click **Component**.
2. Expand the **Sample data** section.
3. Locate **Automobile price data (Raw)**.
4. **Drag** it onto the pipeline canvas.

>  **Screenshot Checkpoint E1:** Capture the Designer canvas showing the **Automobile price data (Raw)** component placed on the canvas.

### Step E.2 – Visualize the Dataset

Before processing the data, examine it to understand its structure.

1. **Right-click** the **Automobile price data (Raw)** component on the canvas.
2. Select **Preview Data** from the context menu.
3. A data preview window opens. Click through the different columns to inspect the data.

Observe the following:
- The dataset contains **205 rows** (one per automobile) and **26 columns** (features)
- The target column is **`price`** — this is what the model will learn to predict
- Notice that the **`normalized-losses`** column contains many blank/missing values — you will handle this in the next step
- Other columns with missing values include `bore`, `stroke`, `horsepower`, and `peak-rpm`

4. Close the preview window.

>  **Screenshot Checkpoint E2:** Capture the data preview window showing the dataset columns and at least a few rows, with the `normalized-losses` column visible.

---

## Part F – Prepare Data (Clean and Transform)

Real-world datasets are never perfectly clean. This part walks through the two preprocessing steps required before training can begin.

### Step F.1 – Add the "Select Columns in Dataset" Component

The `normalized-losses` column has too many missing values to be useful — it will be excluded entirely.

1. In the component palette, click **Component** and search for **Select Columns in Dataset**.
2. Drag the **Select Columns in Dataset** component onto the canvas, below the dataset component.
3. **Connect** the two components: drag from the **output port** (small circle at the bottom) of the **Automobile price data (Raw)** dataset to the **input port** (small circle at the top) of **Select Columns in Dataset**.

> **Tip:** In the Designer, data flows from top to bottom. An output port feeds into an input port of the component below it. This connection defines the pipeline's execution order.

4. **Click** on the **Select Columns in Dataset** component to select it.
5. In the settings panel on the right, click the **arrow icon** (or double-click the component) to open the details pane.
6. Click **Edit column**.
7. In the column selector dialog:
   - From the first drop-down, select **Include** → **All columns** (to start by including everything)
   - Click **+** to add a second rule
   - Set the second rule to: **Exclude** → **Column names**
   - In the text box, type exactly: `normalized-losses`
8. Click **Save** to close the column selector.
9. In the component details pane, expand **Node information**.
10. In the **Comment** field, type: `Exclude normalized losses column`

> Comments appear on the canvas as labels. Use them to document your pipeline, just as you would comment code.

>  **Screenshot Checkpoint F1:** Capture the column selector dialog showing the **Include All columns** rule AND the **Exclude normalized-losses** rule configured.

---

### Step F.2 – Add the "Clean Missing Data" Component

After excluding `normalized-losses`, other columns still contain missing values. Use **Clean Missing Data** to handle them.

1. In the component palette, search for **Clean Missing Data**.
2. Drag it onto the canvas, below the **Select Columns in Dataset** component.
3. **Connect** the output port of **Select Columns in Dataset** to the input port of **Clean Missing Data**.
4. Click on **Clean Missing Data** to open its details pane.
5. Click **Edit column**.
6. In the dialog, set the drop-down to **Include** → **All columns**. Click **Save**.
7. Under **Cleaning mode**, select **Remove entire row**.

> **Why remove entire rows?** This strategy drops any row where any column has a missing value. While this reduces the dataset size slightly, it ensures the training algorithm receives only complete, well-formed records. This is the safest strategy for a beginner pipeline.

8. In the **Node information** section, add the comment: `Remove rows with missing values`

>  **Screenshot Checkpoint F2:** Capture the Designer canvas at this stage showing the three connected components: **Automobile price data (Raw)** → **Select Columns in Dataset** → **Clean Missing Data**.

---

### Step F.3 – Add the "Split Data" Component

Before training, the dataset must be divided into a **training set** (used to teach the model) and a **test set** (used to evaluate the model on unseen data).

1. In the component palette, search for **Split Data**.
2. Drag **Split Data** onto the canvas below **Clean Missing Data**.
3. Connect the **left output port** of **Clean Missing Data** to the input port of **Split Data**.

>  **Important:** The **Clean Missing Data** component has **two output ports**:
> - **Left port** → cleaned data (rows kept)
> - **Right port** → discarded data (rows removed)
>
> You must connect the **left port** to **Split Data**. Connecting the right port would pass only the removed/discarded rows downstream.

4. Click on **Split Data** to open its details pane.
5. Set **Fraction of rows in the first output dataset** to `0.7`

> This creates a 70/30 split: 70% of rows go to training (left output port), and 30% go to testing (right output port). This is a standard starting split ratio for regression problems with moderate-sized datasets.

6. Leave **Randomized split** checked and **Random seed** at `0`.
7. Add the comment: `70% training / 30% test split`

---

## Part G – Train the Machine Learning Model

With clean, split data ready, you can now set up the model training components.

### Step G.1 – Add the "Linear Regression" Algorithm Component

1. In the component palette, search for **Linear Regression**.
2. Drag the **Linear Regression** component onto the canvas. Place it to the **left side** of the canvas, beside the **Split Data** component (not directly below it — it will connect sideways).

> **Why Linear Regression?** Because `price` is a continuous numeric value, regression is the correct algorithm family. Linear Regression assumes a linear relationship between input features and the target. While more sophisticated algorithms exist, Linear Regression is the ideal starting model for learning because its results are interpretable.

---

### Step G.2 – Add the "Train Model" Component

1. In the component palette, search for **Train Model**.
2. Drag the **Train Model** component onto the canvas, below and between the **Linear Regression** and **Split Data** components.
3. Make the following two connections:
   - Connect the **output** of **Linear Regression** → to the **left input port** of **Train Model** (the algorithm port)
   - Connect the **left output port** of **Split Data** (the training 70% data) → to the **right input port** of **Train Model** (the dataset port)

> **Important:** Ensure the **left output port** of **Split Data** (training set) connects to **Train Model**, not the right port (test set). Accidentally training on the test set would invalidate your model evaluation.

4. Click on **Train Model** to open its details pane.
5. Click **Edit column** next to the **Label column** field.
6. In the dialog, set the drop-down to **Column names** and type: `price`

> **Critical:** Type `price` in **all lowercase** with no extra spaces. The column name is case-sensitive. An error will occur if the value does not exactly match the column name in the dataset.

7. Click **Save**.
8. Add the comment: `Train linear regression on 70% of data, predicting price`

>  **Screenshot Checkpoint G1:** Capture the Designer canvas showing **Linear Regression**, **Split Data**, and **Train Model** all connected correctly with visible connection lines between their ports.

---

## Part H – Score and Evaluate the Model

After training, you must test the model on the 30% of data it has never seen, then measure how well it performed.

### Step H.1 – Add the "Score Model" Component

1. In the component palette, search for **Score Model**.
2. Drag **Score Model** onto the canvas below **Train Model**.
3. Make the following two connections:
   - Connect the **output** of **Train Model** → to the **left input port** of **Score Model** (the trained model port)
   - Connect the **right output port** of **Split Data** (the test 30% data) → to the **right input port** of **Score Model** (the dataset port)

> The Score Model component applies the trained model to each row in the test set and generates a **Scored Labels** column — the model's predicted price for each automobile.

---

### Step H.2 – Add the "Evaluate Model" Component

1. In the component palette, search for **Evaluate Model**.
2. Drag **Evaluate Model** onto the canvas below **Score Model**.
3. Connect the **output** of **Score Model** → to the **left input port** of **Evaluate Model**.

> The Evaluate Model component compares the **Scored Labels** (predictions) against the **actual prices** in the test set and computes the performance statistics you will review in Part J.

Your completed pipeline should now have **7 connected components** flowing from top to bottom.

>  **Screenshot Checkpoint H1:** Capture the **final complete pipeline** on the Designer canvas showing all 7 components connected in a single flow from Automobile price data (Raw) at the top to Evaluate Model at the bottom. All connection lines must be visible.

---

## Part I – Submit the Pipeline Job

### Step I.1 – Open the Submit Wizard

1. Click the **Configure & Submit** button in the top-right corner of the Designer canvas.
2. A multi-step submission wizard opens.

---

### Step I.2 – Basics Tab

Configure the experiment and job identity:

| Setting | Value |
|---|---|
| Experiment name | Select **Create new** → type `train-regression-designer-ml` |
| Job display name | `Lab1-Automobile-Price-Regression` *(optional but recommended)* |
| Job description | `Beginner regression pipeline – CST8921 Lab 1` *(optional)* |

Click **Next**.

---

### Step I.3 – Inputs & Outputs Tab

- This page allows you to assign values to any pipeline-level inputs and outputs.
- Because no inputs or outputs were promoted to the pipeline level in this build, **this page is empty**.
- Click **Next** without making any changes.

---

### Step I.4 – Runtime Settings Tab

This is the most critical step — it assigns compute resources to your pipeline.

1. Under **Select compute type**, choose **Compute instance**.
2. Under **Select Azure ML compute instance**, select `lab1-compute` from the drop-down.
3. Leave **Default datastore** at the workspace default.

> **Why does this matter?** Without a compute target, AML does not know where to run your pipeline steps. Every component in the pipeline will use this compute instance unless a component-level override is set. Setting a workspace-level default is the simplest approach for single-instance pipelines.

>  **Screenshot Checkpoint I1:** Capture the **Runtime settings** tab showing `lab1-compute` selected as the compute instance, before clicking Review + Submit.

Click **Next**.

---

### Step I.5 – Review + Submit Tab

1. Review all settings in the summary view.
2. Click **Submit**.
3. After submission, a banner appears at the top of the canvas with a link to the job. Click the link to open the **Job detail page**.

> **Note:** The wizard saves your last configuration. Future pipeline submissions in the same workspace will pre-populate these settings.

---

## Part J – Review Job Results

### Step J.1 – Navigate to the Jobs Page

1. In the left navigation pane, under **Assets**, click **Jobs**.
2. Your job appears under the experiment `train-regression-designer-ml`.
3. Click the job name to open the **Job detail page**.

---

### Step J.2 – Monitor Pipeline Progress

The job detail page displays a visual pipeline graph with real-time status indicators:

| Colour | Meaning |
|---|---|
| 🔵 Blue spinning | Component is currently running |
| ✅ Green checkmark | Component completed successfully |
| 🔴 Red X | Component failed — click to view error logs |
| ⬜ Grey | Component is queued, waiting for upstream components |

- The pipeline runs **sequentially** from top to bottom
- Total runtime is typically **8–15 minutes** on Standard_DS11_v2
- If a component fails, click on it and select **Logs** in the right panel to diagnose the error

> 📸 **Screenshot Checkpoint J1:** Capture the Job detail page showing the pipeline graph with at least some components showing **green (completed)** status, including the job status indicator at the top.

---

### Step J.3 – View Scored Labels (Predictions)

After the job completes:

1. In the pipeline graph on the job detail page, **right-click** the **Score Model** component.
2. Select **Preview data** → **Scored dataset**.
3. A data table opens. Scroll right to find the **Scored Labels** column — these are the model's **predicted prices**.

Observe that the Scored Labels are in the same range as the actual `price` column values. A good model will have Scored Labels close to the actual prices.

> **Screenshot Checkpoint J2:** Capture the **Score Model** preview output showing the data table with the **Scored Labels** column and the original **price** column both visible.

---

### Step J.4 – View Evaluation Metrics

1. In the pipeline graph, **right-click** the **Evaluate Model** component.
2. Select **Preview data** → **Evaluation results**.
3. The evaluation metrics table appears.

Understand each metric as described by Microsoft:

| Metric | What It Measures | Ideal Value |
|---|---|---|
| **Mean Absolute Error (MAE)** | The average of the absolute differences between predicted and actual prices. Expressed in the same unit as price (dollars). | Lower is better |
| **Root Mean Squared Error (RMSE)** | The square root of the average squared differences. Penalises large errors more heavily than MAE. | Lower is better |
| **Relative Absolute Error** | Average absolute errors relative to the absolute difference between actual values and their mean. | Lower is better |
| **Relative Squared Error** | Average squared errors relative to the squared difference between actual values and their mean. | Lower is better |
| **Coefficient of Determination (R²)** | Proportion of variance in the actual prices explained by the model. Ranges from 0 to 1. | Closer to 1.0 is better |

> **Screenshot Checkpoint J3:** Capture the **Evaluate Model** preview output showing the full metrics table with all five metric values visible.

---

### Step J.5 – Analysis Questions (Answer in Lab Report)

Answer the following questions using the actual numbers from your pipeline run:

1. What **R² value** did your model achieve? Using the scale below, how would you classify your model's performance?
   - R² > 0.90 → Excellent fit
   - R² 0.75–0.90 → Good fit
   - R² 0.50–0.75 → Moderate fit
   - R² < 0.50 → Poor fit

2. What is your **MAE** value? In plain language, on average, how many dollars is your model's price prediction off from the actual car price?

3. Find **one specific row** in the Score Model output. Record the actual `price` and the `Scored Label` (predicted price). Calculate the **percentage error** for that row:
   ```
   Percentage Error = |Actual Price – Predicted Price| / Actual Price × 100%
   ```

4. In the pipeline, you used a **70/30 train/test split**. Why is it essential to evaluate the model on data it was *not* trained on? What problem would arise if you evaluated on the same data used for training?

5. The **Clean Missing Data** component was set to "Remove entire row." What is the trade-off of this approach compared to replacing missing values with the column mean? When might you prefer one approach over the other?

6. The **Select Columns in Dataset** component excluded `normalized-losses`. What would likely happen to model performance if you had kept this column without handling its many missing values?

7. Looking at the Automobile dataset columns, list **three features** you believe most strongly predict `price`, and explain briefly why each might be predictive.

---

## Part K – Clean Up Azure Resources

> ⚠️ **This step is mandatory and will be verified in the lab report.** Resources left running consume Azure for Students credits. Students who exhaust their credits before the end of the semester cannot complete subsequent labs. Always clean up after every lab session.

### Step K.1 – Stop the Compute Instance

1. In AML Studio, navigate to **Manage → Compute → Compute instances**.
2. Locate `lab1-compute`.
3. Click the **Stop** button (square icon) in the row.
4. Wait for the status to change to **Stopped**.

> Stopping the instance halts compute billing but retains configuration. Proceed to delete it entirely in the next step.

### Step K.2 – Delete the Compute Instance

1. With `lab1-compute` in **Stopped** state, click the **Delete** button (trash icon).
2. Confirm the deletion when prompted.
3. Wait for the instance to disappear from the list.

### Step K.3 – Delete the Resource Group (Removes All Lab Resources)

Deleting the resource group removes **all associated resources** — workspace, storage account, key vault, and Application Insights — in a single operation.

1. Return to the **Azure Portal** ([https://portal.azure.com](https://portal.azure.com)).
2. In the search bar, type `Resource groups` and select it.
3. Click on **aml-lab-rg**.
4. Click **Delete resource group** (top menu bar).
5. In the confirmation dialog, type the exact name: `aml-lab-rg`
6. Click **Delete**.
7. Wait for the deletion confirmation notification.

>  **Verification:** Navigate to **Resource groups** again and confirm that **aml-lab-rg** no longer appears in the list.

>  **Screenshot Checkpoint K1:** Capture the Azure Portal Resource Groups list showing that **aml-lab-rg** no longer exists (the confirmation of successful deletion).

---

## Lab Report Requirements

Prepare a professional lab report and submit it as a **PDF or Word document** through the **Assignments tab on Brightspace** before the due date.

### Required Sections and Order

**Section 1 – Cover Page**
- Student Full Name, Student ID Number, Course Code (CST8921), Lab Number (Lab 1), Submission Date

**Section 2 – Introduction** *(150–200 words)*
- In your own words: What is Azure Machine Learning? What is the Designer? Why are these tools relevant to cloud professionals in 2025?

**Section 3 – Lab Walkthrough with Screenshots**

Include every screenshot listed below. Each screenshot must have a **caption of 1–2 sentences** explaining what it shows and its significance.

| # | Checkpoint | Description |
|---|---|---|
| A1 | Workspace deployment completion | |
| A2 | AML workspace overview in Azure Portal | |
| B1 | AML Studio navigation pane (all three sections) | |
| C1 | Compute instance in Running state | |
| D1 | Blank Designer canvas with renamed pipeline title | |
| E1 | Dataset component placed on canvas | |
| E2 | Data preview window (dataset structure) | |
| F1 | Column selector showing Include All + Exclude normalized-losses | |
| F2 | Canvas with first three components connected | |
| G1 | Canvas with Linear Regression, Split Data, Train Model connected | |
| H1 | Complete pipeline (all 7 components connected) | |
| I1 | Runtime settings tab with compute instance selected | |
| J1 | Job detail page showing completed (green) components | |
| J2 | Score Model output with Scored Labels visible | |
| J3 | Evaluate Model metrics table | |
| K1 | Resource group deleted (no longer in list) | |

**Section 4 – Analysis and Reflection** *(300–400 words)*
- Answer all **seven questions** from Step J.5 in full sentences, citing your actual metric values from the pipeline run.

**Section 5 – Conclusion** *(100–150 words)*
- Summarise what you learned about building an ML pipeline in Azure.
- Describe one real-world industry scenario (outside automotive) where this regression pipeline approach could solve a business problem.

**Section 6 – Challenges Encountered** *(optional, ~100 words)*
- Briefly describe any errors or unexpected results you encountered and how you resolved them. This section demonstrates professional troubleshooting skills.

---

## Grading Rubric

| Criteria | Points |
|---|---|
| Workspace created with correct settings (region, resource group, workspace name) | 5 |
| Compute instance created with correct VM size and shown in Running state | 5 |
| All 16 required screenshots present with captions | 32 |
| Pipeline built with all 7 components correctly connected (evidenced by screenshots) | 15 |
| All 7 analysis questions answered correctly with actual metric values cited | 28 |
| Introduction and Conclusion sections are substantive and demonstrate understanding | 10 |
| Report is professionally formatted, well-organised, and free of major errors | 5 |
| **Total** | **100** |

---

## Helpful References

| Resource | URL |
|---|---|
| **Tutorial Part 1 – Train (Latest Microsoft Docs)** | https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-train-score?view=azureml-api-1 |
| **Tutorial Part 2 – Deploy Model** | https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-deploy?view=azureml-api-1 |
| **What Is Azure Machine Learning Designer?** | https://learn.microsoft.com/en-us/azure/machine-learning/concept-designer |
| **Create Workspace Resources (Quickstart)** | https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?view=azureml-api-2 |
| **Evaluate Model Component Reference** | https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/evaluate-model |
| **Linear Regression Component Reference** | https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/linear-regression |
| **Clean Missing Data Component Reference** | https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/clean-missing-data |
| **Azure ML Studio (Sign In)** | https://ml.azure.com |
| **Azure Portal (Sign In)** | https://portal.azure.com |
| **SDK v2 Migration Guide (Future Reference)** | https://learn.microsoft.com/en-us/azure/machine-learning/migrate-to-v2 |

---

## Academic Integrity Notice

- This is an **individual** lab. All work and all screenshots must be from **your own** Azure subscription.
- Your Azure Portal screenshots must show your **student email address** (visible in the top-right corner of the portal).
- Sharing screenshots or lab reports between students constitutes academic dishonesty.
- If you encounter subscription or permission issues that prevent you from completing the lab, **email your professor immediately** with a screenshot of the error and your subscription details. Do not wait until the deadline.

---

*CST8921 – Cloud Industry Trends | Lab 1 | Azure Machine Learning Designer*
*Algonquin College School of Advanced Technology*
*References aligned with Microsoft Learn documentation – last verified May 2025*
