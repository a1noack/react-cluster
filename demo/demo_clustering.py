import numpy as np
import os
import pandas as pd
import streamlit as st

main_dir = os.path.join(os.getcwd(), 'demo')

def main():
    st.set_page_config(layout='wide')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 210px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 210px;
            margin-left: -500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # select attack
    # st.markdown(f'''
    # <style>
    #     section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}
    #     section[data-testid="stSidebar"] .css-1d391kg {{width: 14rem;}}
    # </style>
    # ''',unsafe_allow_html=True)

    st.sidebar.write('Select Attack')
    attacks_full_names = {}
    attacks = ['Clean', 'DeepWordBug', 'Faster Gen. (Unseen)', 'Genetic (Unseen)', 'HotFlip',
               'Iga Wang (Unseen)', 'Pruthi', 'TextBugger', 'Show all']
    attacks_filename = {'Clean': 'clean', 'DeepWordBug': 'deepwordbug', 'Faster Gen. (Unseen)': 'faster_genetic',
                        'Genetic (Unseen)': 'genetic', 'HotFlip': 'hotflip', 'Iga Wang (Unseen)': 'iga_wang',
                        'Pruthi': 'pruthi', 'TextBugger': 'textbugger', 'Show all': None}
    attack_name = attacks_filename[st.sidebar.selectbox('Attack Name', attacks)]

    # select groups
    attacks_to_groups = {
        'clean': [5, 11, 22, 32],
        'deepwordbug': [18, 19, 34, 36],
        'faster_genetic': [183, 210, 338, 584],
        'genetic': [4, 9, 12, 14 ],
        'hotflip': [1, 3, 8, 10],
        'iga_wang': [13, 15, 17, 24],
        'pruthi': [0, 2, 6, 21],
        'textbugger': [7, 29, 35, 39]
    }

    group_number = None
    if attack_name:
        st.sidebar.write('Select Group')
        group_number = st.sidebar.selectbox('Group Number', attacks_to_groups[attack_name])

    texts_path = None

    st.markdown('### Clustering groups of successful attacks in the Siamese Net embedding space')
    # st.header('<span style="color:#008B8B;">  </span>', '')
    if attack_name is not None and group_number is not None:
        attack_path = os.path.join(main_dir, 'texts_to_plots', str(attack_name)+'_'+str(group_number))
        image_path = os.path.join(attack_path, 'plot_X.jpg')
        texts_path = os.path.join(attack_path, 'texts.npy')
        orig_texts_path = os.path.join(attack_path, 'orig_texts.npy')
        predictions_path = os.path.join(attack_path, 'predictions.npy')
    else:
        image_path = os.path.join(main_dir, 'texts_to_plots', 'all_plot.jpg')

    st.image(image_path, caption=None, channels="RGB")

    if texts_path:
        st.markdown('##### Samples in the group indicated by the black "X": ')
        texts_npy = np.load(texts_path)
        orig_texts_npy = np.load(orig_texts_path)
        predictions_npy = (np.load(predictions_path) - 1) * -1
        if attack_name != 'clean':
            predictions_npy = np.where(predictions_npy == 1, '+', '-')
        else:
            predictions_npy = np.where(predictions_npy == 1, '-', '+')


        if attack_name != 'clean':
            df = pd.DataFrame(texts_npy, columns=['Perturbed text'])
            df['Original text'] = orig_texts_npy
            df['Predicted sentiment'] = predictions_npy
            df = df[['Original text', 'Perturbed text', 'Predicted sentiment']]
        else:
            df = pd.DataFrame(texts_npy, columns=['Original/perturbed text'])
            df['Predicted sentiment'] = predictions_npy
            df = df[['Original/perturbed text', 'Predicted sentiment']]
        st.table(df)


if __name__ == '__main__':
    main()
