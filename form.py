import streamlit as st
import pandas as pd
import pdfplumber
import io
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import json

# Configuração da página para usar toda a largura disponível
st.set_page_config(layout="wide")

# Carregar as credenciais do arquivo JSON
with open('config.json', 'r') as f:
    config = json.load(f)

endpoint = config['AZURE_ENDPOINT']
key = config['AZURE_KEY']
model_id = config['MODEL_ID']

# Inicializar o cliente do Azure Form Recognizer
document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def process_image(image_bytes):
    poller = document_analysis_client.begin_analyze_document(model_id=model_id, document=image_bytes)
    result = poller.result()
    document_data = []

    for document in result.documents:
        for name, field in document.fields.items():
            field_value = field.value if field.value else field.content
            document_data.append({
                "Field Name": name,
                "Field Value": field_value,
                "Confidence": field.confidence
            })

    return pd.DataFrame(document_data)

def main():
    # Adicionando uma imagem na sidebar
    st.sidebar.image("LM4.png", use_column_width=True)
    
    st.title("Extração e Análise de Dados de Documentos PDF")
    
    uploaded_file = st.file_uploader("Carregue o arquivo PDF:", type="pdf")
    
    if st.button("Analisar Documento") and uploaded_file is not None:
        all_pages_df = []

        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= 5:  # Limitando a 5 páginas
                    break
                
                img = page.to_image()
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                page_df = process_image(img_byte_arr)
                
                if not page_df.empty:
                    # Pivotando os dados para a estrutura desejada
                    page_df_pivot = page_df.pivot_table(index=['Field Name'], values='Field Value', aggfunc='first').T
                    page_df_pivot['Confidence'] = page_df['Confidence'].mean()
                    all_pages_df.append(page_df_pivot)
        
        if all_pages_df:
            combined_df = pd.concat(all_pages_df).reset_index(drop=True)
            
            # Ajustando a ordem das colunas conforme solicitado
            cols = list(combined_df.columns)
            cols = [cols[-2]] + [cols[-1]] + cols[:-2]  # Movendo a penúltima coluna para ser a primeira
            combined_df = combined_df[cols]
            combined_df.columns.values[0] = "NOME"  # Renomeando a primeira coluna para "NOME"
            
            st.subheader("Dados Extraídos do Documento PDF:")
            st.dataframe(combined_df)
            
            # Permitir o download do DataFrame como Excel
            towrite = io.BytesIO()
            combined_df.to_excel(towrite, index=False, header=True)
            towrite.seek(0)  # reset pointer
            st.download_button(
                label="Download Excel",
                data=towrite,
                file_name='dados_extraidos.xlsx',
                mime='application/vnd.ms-excel'
            )
        else:
            st.error("Não foi possível extrair dados do documento.")

if __name__ == "__main__":
    main()
