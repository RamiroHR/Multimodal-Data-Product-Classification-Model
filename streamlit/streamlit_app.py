import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import project_tools as pt



st.title("Multimodal Product Data Classification")


st.sidebar.title("Table of contents")
pages = ["Project Description", "Fusion Model", "Performance", "Try it!"]

page = st.sidebar.radio("Go to", pages)


### First page
if page == "Project Description":
    st.markdown("## Enhancing e-commerce")

    st.write(pt.project_applications())   
    st.image("./images/rakuten_e_commerce.png" )

    st.markdown("## ")
    st.write(pt.project_description())
    st.image("./images/Fusion_model_classification_example.png")
    
    # st.markdown('### Data Description')

    # st.dataframe(pt.get_df_head(7))
    # st.write(pt.get_df_shape())
    # st.dataframe(pt.get_df_description())

    # if st.checkbox("Show NA"):
    #     st.dataframe(df.isna().sum())


if page == 'Fusion Model':

    st.write("## ", page)
    st.markdown("# ")
    st.write(pt.model_description()['Fusion'])
    st.image("./images/Fusion_model_schema.png")

    st.markdown("# ")
    st.markdown("##### Headless Image Model:")
    st.markdown(pt.model_description()['hl_image'])

    st.markdown("# ")
    st.markdown("##### Headless Text Model:")
    st.markdown(pt.model_description()['hl_text'])

    st.markdown("# ")
    st.markdown("##### Fusion Head model:")
    st.markdown(pt.model_description()['fusion_head'])


if page == 'Performance':

    st.write("## ", page)
    st.write(pt.performance_description('perf_comparison'))
    st.image("./images/fusion_performance_comparison.png")

    st.markdown("## ")
    st.write(pt.performance_description('confusion_matrix'))
    st.image("./images/fusion_cm.png")

    st.markdown("## ")
    st.write(pt.performance_description('classification_report'))
    st.image("./images/fusion_classification_report.png")



if page == 'Try it!':

    ## Add demo descriptions:
    url = "https://fr.shopping.rakuten.com/"
    st.markdown(pt.demo_instructions() % url)
    st.write("####")        

    ## inputs
    sample_title = st.text_input("**Product Title:**", key="title")

    sample_description = st.text_area("**Product Description:**", 
                                       key="description",
                                       placeholder = '') 
    # sample_description
    # sample_image = None
    sample_image = st.file_uploader('**Product image:**') 
    # st.write(type(sample_image.read()))

    # Button to start the predictions
    run_model = st.button("Get Prediction", type="primary")  

    if run_model: 
        # file_bytes = np.asarray(bytearray(sample_image.read()), dtype=np.uint8) 
        # opencv_image = cv2.imdecode(file_bytes, 1)

        ## 
        # st.write(pt.get_image_hl_model_output(sample_image).shape)
        # st.write(pt.get_text_hl_model_output(sample_title, sample_description).shape)
        with st.spinner('Wait for it...'):        
            sample_pred_codes, sample_pred_classes, pred_confidences = pt.get_predictions(sample_title, sample_description, sample_image)
        
        # @st.cache_data()
        st.markdown("#### Top 3 predictions:")
        for i in range(3):
            # st.markdown(f'* **{np.round(float(pred_confidences[i])*100, 2)} %** confidence of \
            #         being category "**{sample_pred_classes[i]} ({sample_pred_codes[i]})**"')
            st.success(f'**{np.round(float(pred_confidences[i])*100, 2)} %** confidence of \
                    being category "**{sample_pred_classes[i]} ({sample_pred_codes[i]})**"', icon="✅")

        # st.write("Predicted class code:", sample_pred_codes)
        # st.write("Predicted class name:", sample_pred_classes)
        # st.write("Predicted class name:", pred_confidences) 


        # @st.cache_data()
        with st.expander("Product Summary", expanded = True): 
            st.image(sample_image, width = 400)
            st.markdown(f"**Title:** {sample_title}")
            st.markdown(f"**Description:** {sample_description}")
 
        st.markdown("## ")
        with st.expander("Display Confusion Matrix"):
            st.image("./images/fusion_cm.png") #, width=250 
            st.image("./images/code_category.png")


