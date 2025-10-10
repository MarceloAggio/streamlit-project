import streamlit as st
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import io
import json
import os
import imaplib
import email
from email.header import decode_header

# Configuração da página
st.set_page_config(
    page_title="Sistema de Pedidos - Livraria",
    page_icon="📚",
    layout="wide"
)

# Arquivos de configuração
ARQUIVO_CLIENTES = "clientes.json"
ARQUIVO_COLUNAS = "colunas_config.json"
ARQUIVO_PEDIDOS = "pedidos.json"

# Configurações padrão
CLIENTES_PADRAO = {
    "Leitura": {
        "codigo": "LT001",
        "cnpj": "00.000.000/0001-00",
        "email": "contato@leitura.com.br"
    },
    "Catavento": {
        "codigo": "CV002",
        "cnpj": "11.111.111/0001-11",
        "email": "contato@catavento.com.br"
    },
    "Livraria da Vila": {
        "codigo": "LV003",
        "cnpj": "22.222.222/0001-22",
        "email": "contato@livrariadavila.com.br"
    }
}

COLUNAS_PADRAO = {
    "colunas_esperadas": ["codigo", "nome", "quantidade"],
    "colunas_alternativas": {
        "codigo": ["código", "cod", "sku", "isbn", "code"],
        "nome": ["título", "titulo", "livro", "produto", "descricao", "descrição"],
        "quantidade": ["qtd", "qtde", "quant", "qty"]
    }
}

# Funções de gerenciamento de arquivos JSON
def carregar_clientes():
    """Carrega clientes do arquivo JSON ou cria com dados padrão"""
    if os.path.exists(ARQUIVO_CLIENTES):
        try:
            with open(ARQUIVO_CLIENTES, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return CLIENTES_PADRAO
    else:
        salvar_clientes(CLIENTES_PADRAO)
        return CLIENTES_PADRAO

def salvar_clientes(clientes):
    """Salva clientes no arquivo JSON"""
    with open(ARQUIVO_CLIENTES, 'w', encoding='utf-8') as f:
        json.dump(clientes, f, ensure_ascii=False, indent=4)

def carregar_config_colunas():
    """Carrega configuração de colunas do arquivo JSON"""
    if os.path.exists(ARQUIVO_COLUNAS):
        try:
            with open(ARQUIVO_COLUNAS, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return COLUNAS_PADRAO
    else:
        salvar_config_colunas(COLUNAS_PADRAO)
        return COLUNAS_PADRAO

def salvar_config_colunas(config):
    """Salva configuração de colunas no arquivo JSON"""
    with open(ARQUIVO_COLUNAS, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def carregar_pedidos():
    """Carrega pedidos salvos"""
    if os.path.exists(ARQUIVO_PEDIDOS):
        try:
            with open(ARQUIVO_PEDIDOS, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def salvar_pedidos(pedidos):
    """Salva pedidos no arquivo JSON"""
    with open(ARQUIVO_PEDIDOS, 'w', encoding='utf-8') as f:
        json.dump(pedidos, f, ensure_ascii=False, indent=4)

# Inicializar session_state
if 'pedidos' not in st.session_state:
    st.session_state.pedidos = carregar_pedidos()
if 'config' not in st.session_state:
    st.session_state.config = {
        'email_faturamento': '',
        'email_remetente': '',
        'senha_email': '',
        'smtp_servidor': 'smtp.gmail.com',
        'smtp_porta': 587
    }
if 'clientes' not in st.session_state:
    st.session_state.clientes = carregar_clientes()
if 'config_colunas' not in st.session_state:
    st.session_state.config_colunas = carregar_config_colunas()
if 'emails_recebidos' not in st.session_state:
    st.session_state.emails_recebidos = []

# Categorias de produtos
CATEGORIAS_PRODUTO = {
    'Bíblia': 'ISENTO',
    'Livro Religioso': 'ISENTO',
    'Livro Geral': 'TRIBUTADO'
}

def ler_emails_gmail_imap(email_usuario, senha_app, max_results=50):
    """
    Lê emails do Gmail usando IMAP - Muito mais simples!
    
    Args:
        email_usuario: seu email do Gmail
        senha_app: senha de aplicativo do Gmail
        max_results: número máximo de emails para buscar
    """
    try:
        st.info("🔌 Conectando ao Gmail via IMAP...")
        
        # Conectar ao servidor IMAP do Gmail
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_usuario, senha_app)
        mail.select("INBOX")
        
        # Buscar emails com "pedido" no assunto ou corpo
        st.info("🔍 Buscando emails com pedidos...")
        status, messages = mail.search(None, '(OR SUBJECT "pedido" BODY "pedido")')
        
        if status != "OK":
            st.error("❌ Erro ao buscar emails")
            return []
        
        # Pegar os IDs dos últimos N emails
        email_ids = messages[0].split()
        email_ids = email_ids[-max_results:] if len(email_ids) > max_results else email_ids
        
        emails_pedidos = []
        total = len(email_ids)
        
        if total == 0:
            st.warning("📭 Nenhum email com 'pedido' encontrado")
            mail.close()
            mail.logout()
            return []
        
        # Barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, email_id in enumerate(email_ids):
            try:
                # Atualizar progresso
                progress = (idx + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"📧 Processando email {idx + 1} de {total}...")
                
                # Buscar o email
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        # Parse do email
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Extrair assunto
                        subject = msg.get("Subject", "")
                        if subject:
                            decoded_subject = decode_header(subject)[0]
                            if isinstance(decoded_subject[0], bytes):
                                subject = decoded_subject[0].decode(decoded_subject[1] or 'utf-8', errors='ignore')
                            else:
                                subject = decoded_subject[0]
                        
                        # Extrair remetente
                        from_email = msg.get("From", "")
                        
                        # Extrair data
                        date = msg.get("Date", "")
                        
                        # Extrair corpo e anexos
                        corpo = ""
                        anexos = []
                        
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                content_disposition = str(part.get("Content-Disposition", ""))
                                
                                # Corpo do email
                                if content_type == "text/plain" and "attachment" not in content_disposition:
                                    try:
                                        payload = part.get_payload(decode=True)
                                        if payload:
                                            corpo += payload.decode('utf-8', errors='ignore')
                                    except:
                                        pass
                                
                                # Anexos (apenas planilhas)
                                if "attachment" in content_disposition:
                                    filename = part.get_filename()
                                    if filename and filename.lower().endswith(('.xlsx', '.csv', '.xls')):
                                        # Decodificar nome do arquivo se necessário
                                        if filename:
                                            decoded_filename = decode_header(filename)[0]
                                            if isinstance(decoded_filename[0], bytes):
                                                filename = decoded_filename[0].decode(decoded_filename[1] or 'utf-8', errors='ignore')
                                            else:
                                                filename = decoded_filename[0]
                                        
                                        # Obter dados do anexo
                                        anexo_data = part.get_payload(decode=True)
                                        if anexo_data:
                                            anexos.append({
                                                'filename': filename,
                                                'data': anexo_data
                                            })
                        else:
                            # Email não é multipart
                            try:
                                payload = msg.get_payload(decode=True)
                                if payload:
                                    corpo = payload.decode('utf-8', errors='ignore')
                            except:
                                pass
                        
                        # Adicionar à lista
                        emails_pedidos.append({
                            'id': email_id.decode(),
                            'assunto': subject,
                            'remetente': from_email,
                            'data': date,
                            'corpo': corpo[:500],  # Prévia
                            'corpo_completo': corpo,
                            'anexos': anexos
                        })
            
            except Exception as e:
                continue
        
        # Limpar progresso
        progress_bar.empty()
        status_text.empty()
        
        # Fechar conexão
        mail.close()
        mail.logout()
        
        return emails_pedidos
    
    except imaplib.IMAP4.error as e:
        st.error(f"❌ Erro de autenticação IMAP: {str(e)}")
        st.warning("⚠️ Verifique se você está usando a SENHA DE APLICATIVO (não a senha normal do Gmail)")
        return []
    except Exception as e:
        st.error(f"❌ Erro ao conectar: {str(e)}")
        return []

def carregar_planilha(arquivo, config_colunas):
    """Carrega a planilha do cliente"""
    try:
        if arquivo.name.endswith('.xlsx'):
            df = pd.read_excel(arquivo)
        elif arquivo.name.endswith('.csv'):
            df = pd.read_csv(arquivo, encoding='utf-8')
        else:
            st.error("Formato não suportado. Use .xlsx ou .csv")
            return None
        
        # Padronizar nomes das colunas
        df.columns = df.columns.str.lower().str.strip()
        
        st.info(f"📋 Colunas detectadas na planilha: {', '.join(df.columns.tolist())}")
        
        # Verificar e mapear colunas
        colunas_mapeadas = {}
        for col_padrao in config_colunas['colunas_esperadas']:
            if col_padrao in df.columns:
                colunas_mapeadas[col_padrao] = col_padrao
                continue
            
            encontrou = False
            for alternativa in config_colunas['colunas_alternativas'].get(col_padrao, []):
                if alternativa in df.columns:
                    df.rename(columns={alternativa: col_padrao}, inplace=True)
                    colunas_mapeadas[col_padrao] = alternativa
                    encontrou = True
                    break
            
            if not encontrou:
                st.warning(f"⚠️ Coluna '{col_padrao}' não encontrada na planilha!")
                return None
        
        if colunas_mapeadas:
            st.success(f"✅ Colunas mapeadas: {colunas_mapeadas}")
        
        df = df[config_colunas['colunas_esperadas']].copy()
        df = df.dropna(subset=config_colunas['colunas_esperadas'])
        df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar planilha: {str(e)}")
        return None

def carregar_planilha_de_bytes(dados_bytes, nome_arquivo, config_colunas):
    """Carrega planilha a partir de bytes (anexo de email)"""
    try:
        if nome_arquivo.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(dados_bytes))
        elif nome_arquivo.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(dados_bytes), encoding='utf-8')
        else:
            return None
        
        df.columns = df.columns.str.lower().str.strip()
        
        for col_padrao in config_colunas['colunas_esperadas']:
            if col_padrao in df.columns:
                continue
            
            for alternativa in config_colunas['colunas_alternativas'].get(col_padrao, []):
                if alternativa in df.columns:
                    df.rename(columns={alternativa: col_padrao}, inplace=True)
                    break
        
        df = df[config_colunas['colunas_esperadas']].copy()
        df = df.dropna(subset=config_colunas['colunas_esperadas'])
        df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        return None

def enviar_email_faturamento(cliente, df_pedido, dados_config):
    """Envia email para o faturamento"""
    try:
        if not all([dados_config['email_remetente'], 
                   dados_config['senha_email'],
                   dados_config['email_faturamento']]):
            st.error("❌ Configurações de email incompletas!")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = dados_config['email_remetente']
        msg['To'] = dados_config['email_faturamento']
        msg['Subject'] = f"Pedido - {cliente} - {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        
        cliente_info = st.session_state.clientes[cliente]
        
        corpo = f"""
Olá equipe de faturamento,

Segue novo pedido para processamento:

Cliente: {cliente}
Código Cliente: {cliente_info['codigo']}
CNPJ: {cliente_info['cnpj']}
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Total de itens: {len(df_pedido)}
Quantidade total: {df_pedido['quantidade'].sum()}

Por favor, processar no sistema Oracle e retornar com:
- Nota Fiscal
- Itens não atendidos (se houver)
- Previsão de entrega

Planilha em anexo.

Atenciosamente,
Sistema de Pedidos
        """
        
        msg.attach(MIMEText(corpo, 'plain', 'utf-8'))
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_pedido.to_excel(writer, index=False, sheet_name='Pedido')
        output.seek(0)
        
        part = MIMEBase('application', 'vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        part.set_payload(output.read())
        encoders.encode_base64(part)
        
        nome_arquivo = f'pedido_{cliente.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
        part.add_header('Content-Disposition', f'attachment; filename={nome_arquivo}')
        msg.attach(part)
        
        servidor = dados_config.get('smtp_servidor', 'smtp.gmail.com')
        porta = dados_config.get('smtp_porta', 587)
        
        server = smtplib.SMTP(servidor, porta)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(dados_config['email_remetente'], dados_config['senha_email'])
        
        texto = msg.as_string()
        server.sendmail(dados_config['email_remetente'], dados_config['email_faturamento'], texto)
        server.quit()
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        st.error("❌ Erro de autenticação. Verifique email e senha (use senha de aplicativo para Gmail).")
        return False
    except smtplib.SMTPException as e:
        st.error(f"❌ Erro SMTP: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Erro ao enviar email: {str(e)}")
        return False

def enviar_retorno_cliente(cliente, nota_fiscal, itens_nao_atendidos, dados_config):
    """Envia retorno para o cliente"""
    try:
        if not all([dados_config['email_remetente'], dados_config['senha_email']]):
            st.error("❌ Configurações de email incompletas!")
            return False
        
        cliente_info = st.session_state.clientes[cliente]
        
        msg = MIMEMultipart()
        msg['From'] = dados_config['email_remetente']
        msg['To'] = cliente_info['email']
        msg['Subject'] = f"Retorno Pedido - {datetime.now().strftime('%d/%m/%Y')}"
        
        corpo = f"""
Prezado(a) {cliente},

Segue retorno do seu pedido:

Nota Fiscal: {nota_fiscal}
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""
        
        if itens_nao_atendidos and len(itens_nao_atendidos) > 0:
            corpo += f"\n\nItens não atendidos ({len(itens_nao_atendidos)}):\n"
            for item in itens_nao_atendidos:
                corpo += f"- {item['codigo']} - {item['nome']} (Qtd: {item['quantidade']}) - Motivo: {item['motivo']}\n"
        else:
            corpo += "\n\nTodos os itens foram atendidos! ✓"
        
        corpo += """

Qualquer dúvida, estamos à disposição.

Atenciosamente,
Equipe Comercial
        """
        
        msg.attach(MIMEText(corpo, 'plain', 'utf-8'))
        
        servidor = dados_config.get('smtp_servidor', 'smtp.gmail.com')
        porta = dados_config.get('smtp_porta', 587)
        
        server = smtplib.SMTP(servidor, porta)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(dados_config['email_remetente'], dados_config['senha_email'])
        
        texto = msg.as_string()
        server.sendmail(dados_config['email_remetente'], cliente_info['email'], texto)
        server.quit()
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        st.error("❌ Erro de autenticação. Verifique email e senha.")
        return False
    except Exception as e:
        st.error(f"❌ Erro ao enviar retorno: {str(e)}")
        return False

# Interface principal
st.title("📚 Sistema de Gerenciamento de Pedidos")
st.markdown("---")

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    with st.expander("📧 Configurar Emails", expanded=False):
        st.session_state.config['email_faturamento'] = st.text_input(
            "Email Faturamento",
            value=st.session_state.config['email_faturamento'],
            placeholder="faturamento@empresa.com.br"
        )
        st.session_state.config['email_remetente'] = st.text_input(
            "Seu Email (Gmail)",
            value=st.session_state.config['email_remetente'],
            placeholder="seu.email@gmail.com"
        )
        st.session_state.config['senha_email'] = st.text_input(
            "Senha de Aplicativo Gmail",
            value=st.session_state.config['senha_email'],
            type="password",
            help="Use senha de aplicativo, não a senha normal"
        )
        
        st.divider()
        st.caption("**Configurações SMTP (Envio)**")
        st.session_state.config['smtp_servidor'] = st.text_input(
            "Servidor SMTP",
            value=st.session_state.config.get('smtp_servidor', 'smtp.gmail.com')
        )
        st.session_state.config['smtp_porta'] = st.number_input(
            "Porta SMTP",
            value=st.session_state.config.get('smtp_porta', 587),
            min_value=1,
            max_value=65535
        )
        
        st.caption("⚠️ Para Gmail, use uma senha de aplicativo")
        if st.button("📖 Como criar senha de aplicativo?"):
            st.info("""
            **Passo a passo:**
            1. Acesse: myaccount.google.com
            2. Vá em "Segurança"
            3. Ative "Verificação em duas etapas" (obrigatório)
            4. Role até "Senhas de app"
            5. Selecione "Email" → "Outro dispositivo"
            6. Dê um nome (ex: "Sistema Pedidos")
            7. Copie a senha gerada (16 caracteres)
            8. Cole no campo acima
            
            ⚠️ A senha de aplicativo é DIFERENTE da sua senha do Gmail!
            """)
    
    with st.expander("📬 Leitura de Emails (IMAP)", expanded=False):
        st.info("""
        **Como funciona:**
        - Usa IMAP para ler emails do Gmail
        - Busca automaticamente emails com "pedido"
        - Extrai anexos Excel/CSV
        - Muito mais simples que OAuth!
        """)
        
        st.success("✅ Use a mesma senha de aplicativo configurada acima")
        
        st.caption("""
        **Requisitos:**
        - Email do Gmail configurado
        - Senha de aplicativo configurada
        - IMAP habilitado no Gmail (geralmente já está)
        """)
    
    with st.expander("👥 Gerenciar Clientes", expanded=False):
        st.subheader("Adicionar Novo Cliente")
        
        novo_nome = st.text_input("Nome do Cliente", key="novo_cliente_nome")
        novo_codigo = st.text_input("Código", key="novo_cliente_codigo")
        novo_cnpj = st.text_input("CNPJ", key="novo_cliente_cnpj")
        novo_email = st.text_input("Email", key="novo_cliente_email")
        
        if st.button("➕ Adicionar Cliente"):
            if all([novo_nome, novo_codigo, novo_cnpj, novo_email]):
                st.session_state.clientes[novo_nome] = {
                    "codigo": novo_codigo,
                    "cnpj": novo_cnpj,
                    "email": novo_email
                }
                salvar_clientes(st.session_state.clientes)
                st.success(f"✅ Cliente '{novo_nome}' adicionado!")
                st.rerun()
            else:
                st.error("❌ Preencha todos os campos!")
        
        st.divider()
        st.subheader("Clientes Cadastrados")
        
        for nome_cliente in list(st.session_state.clientes.keys()):
            with st.container():
                st.write(f"**{nome_cliente}**")
                st.caption(f"Código: {st.session_state.clientes[nome_cliente]['codigo']}")
                if st.button(f"🗑️ Excluir", key=f"del_{nome_cliente}"):
                    del st.session_state.clientes[nome_cliente]
                    salvar_clientes(st.session_state.clientes)
                    st.success(f"Cliente '{nome_cliente}' excluído!")
                    st.rerun()
                st.divider()
    
    with st.expander("📋 Configurar Colunas CSV", expanded=False):
        st.subheader("Colunas Esperadas")
        
        colunas_esperadas_texto = st.text_area(
            "Colunas principais (uma por linha)",
            value="\n".join(st.session_state.config_colunas['colunas_esperadas']),
            height=100
        )
        
        st.subheader("Alternativas para 'codigo'")
        alt_codigo = st.text_area(
            "Nomes alternativos (um por linha)",
            value="\n".join(st.session_state.config_colunas['colunas_alternativas'].get('codigo', [])),
            height=100
        )
        
        st.subheader("Alternativas para 'nome'")
        alt_nome = st.text_area(
            "Nomes alternativos (um por linha)",
            value="\n".join(st.session_state.config_colunas['colunas_alternativas'].get('nome', [])),
            height=100
        )
        
        st.subheader("Alternativas para 'quantidade'")
        alt_quantidade = st.text_area(
            "Nomes alternativos (um por linha)",
            value="\n".join(st.session_state.config_colunas['colunas_alternativas'].get('quantidade', [])),
            height=100
        )
        
        if st.button("💾 Salvar Configuração de Colunas"):
            nova_config = {
                'colunas_esperadas': [c.strip() for c in colunas_esperadas_texto.split('\n') if c.strip()],
                'colunas_alternativas': {
                    'codigo': [c.strip() for c in alt_codigo.split('\n') if c.strip()],
                    'nome': [c.strip() for c in alt_nome.split('\n') if c.strip()],
                    'quantidade': [c.strip() for c in alt_quantidade.split('\n') if c.strip()]
                }
            }
            st.session_state.config_colunas = nova_config
            salvar_config_colunas(nova_config)
            st.success("✅ Configuração de colunas salva!")
    
    st.markdown("---")
    st.header("📊 Resumo")
    st.metric("Pedidos Ativos", len([p for p in st.session_state.pedidos if p['status'] != 'Concluído']))
    st.metric("Total de Pedidos", len(st.session_state.pedidos))
    st.metric("Clientes Cadastrados", len(st.session_state.clientes))

# Tabs principais
tab1, tab2, tab3, tab4 = st.tabs(["📧 Emails Recebidos", "📥 Novo Pedido", "📋 Acompanhamento", "✅ Finalizar Pedido"])

# TAB 1 - EMAILS RECEBIDOS
with tab1:
    st.header("📧 Verificar Emails com Pedidos")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💡 Esta aba lê seus emails via IMAP e exibe aqueles que contêm 'pedido' no assunto ou corpo.")
    
    with col2:
        if st.button("🔄 Buscar Emails", type="primary", use_container_width=True):
            # Verificar configurações
            if not st.session_state.config['email_remetente']:
                st.error("❌ Configure seu email na barra lateral!")
            elif not st.session_state.config['senha_email']:
                st.error("❌ Configure a senha de aplicativo na barra lateral!")
            else:
                with st.spinner("Conectando ao Gmail..."):
                    emails = ler_emails_gmail_imap(
                        st.session_state.config['email_remetente'],
                        st.session_state.config['senha_email'],
                        max_results=50
                    )
                    st.session_state.emails_recebidos = emails
                    
                    if emails:
                        st.success(f"✅ {len(emails)} email(s) encontrado(s) com pedidos!")
                    else:
                        st.warning("⚠️ Nenhum email com 'pedido' encontrado.")
    
    st.markdown("---")
    
    if len(st.session_state.emails_recebidos) == 0:
        st.info("📭 Clique em 'Buscar Emails' para verificar sua caixa de entrada.")
    else:
        st.subheader(f"Emails Encontrados ({len(st.session_state.emails_recebidos)})")
        
        for idx, email_info in enumerate(st.session_state.emails_recebidos):
            with st.expander(f"📨 {email_info['assunto']} - {email_info['remetente']}", expanded=(idx==0)):
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.write(f"**De:** {email_info['remetente']}")
                    st.write(f"**Data:** {email_info['data']}")
                
                with col_info2:
                    st.write(f"**Anexos:** {len(email_info['anexos'])} arquivo(s)")
                
                st.markdown("**Prévia do corpo:**")
                st.text_area(
                    "Conteúdo",
                    value=email_info['corpo'],
                    height=150,
                    key=f"corpo_{idx}",
                    disabled=True
                )
                
                # Processar anexos
                if email_info['anexos']:
                    st.markdown("**Planilhas anexadas:**")
                    
                    for anexo_idx, anexo in enumerate(email_info['anexos']):
                        st.write(f"📎 {anexo['filename']}")
                        
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            # Tentar carregar a planilha
                            df_anexo = carregar_planilha_de_bytes(
                                anexo['data'],
                                anexo['filename'],
                                st.session_state.config_colunas
                            )
                            
                            if df_anexo is not None:
                                if st.button(f"👁️ Visualizar", key=f"view_{idx}_{anexo_idx}"):
                                    st.dataframe(df_anexo, use_container_width=True)
                        
                        with col_btn2:
                            # Download do anexo
                            st.download_button(
                                label="📥 Baixar",
                                data=anexo['data'],
                                file_name=anexo['filename'],
                                mime="application/octet-stream",
                                key=f"download_{idx}_{anexo_idx}"
                            )
                        
                        with col_btn3:
                            # Criar pedido a partir do anexo
                            if st.button(f"➕ Criar Pedido", key=f"create_{idx}_{anexo_idx}", type="primary"):
                                if df_anexo is not None:
                                    st.session_state.temp_df = df_anexo
                                    st.session_state.temp_email_info = email_info
                                    st.success("✅ Vá para a aba 'Novo Pedido' para processar!")
                                else:
                                    st.error("❌ Não foi possível processar a planilha.")
                else:
                    st.warning("⚠️ Este email não contém anexos de planilha.")

# TAB 2 - NOVO PEDIDO
with tab2:
    st.header("Criar Novo Pedido")
    
    # Verificar se existe dados temporários de email
    if 'temp_df' in st.session_state and st.session_state.temp_df is not None:
        st.info("📧 Dados carregados de email recebido!")
        
        col_dados1, col_dados2 = st.columns(2)
        with col_dados1:
            st.write("**Email:**", st.session_state.temp_email_info['remetente'])
            st.write("**Assunto:**", st.session_state.temp_email_info['assunto'])
        with col_dados2:
            if st.button("🗑️ Limpar dados do email"):
                del st.session_state.temp_df
                del st.session_state.temp_email_info
                st.rerun()
    
    if len(st.session_state.clientes) == 0:
        st.warning("⚠️ Nenhum cliente cadastrado. Adicione clientes na barra lateral.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            cliente_selecionado = st.selectbox(
                "Selecione o Cliente",
                options=list(st.session_state.clientes.keys())
            )
            
            cliente_info = st.session_state.clientes[cliente_selecionado]
            st.info(f"""
            **Dados do Cliente:**
            - Código: {cliente_info['codigo']}
            - CNPJ: {cliente_info['cnpj']}
            - Email: {cliente_info['email']}
            """)
        
        with col2:
            # Verificar se há dados do email
            if 'temp_df' in st.session_state and st.session_state.temp_df is not None:
                st.success("✅ Planilha carregada do email!")
                usar_dados_email = st.checkbox("Usar dados do email", value=True)
            else:
                usar_dados_email = False
                arquivo = st.file_uploader(
                    "Upload da Planilha do Cliente",
                    type=['xlsx', 'csv'],
                    help="Planilha deve conter as colunas configuradas"
                )
        
        # Determinar qual DataFrame usar
        df_pedido = None
        
        if usar_dados_email and 'temp_df' in st.session_state:
            df_pedido = st.session_state.temp_df
        elif not usar_dados_email and 'arquivo' in locals() and arquivo:
            df_pedido = carregar_planilha(arquivo, st.session_state.config_colunas)
        
        if df_pedido is not None:
            st.success(f"✅ Planilha carregada: {len(df_pedido)} itens")
            
            # Adicionar categoria
            st.subheader("Classificação dos Produtos")
            df_pedido['categoria'] = 'Livro Geral'
            df_pedido['tributacao'] = 'TRIBUTADO'
            
            df_editado = st.data_editor(
                df_pedido,
                column_config={
                    "categoria": st.column_config.SelectboxColumn(
                        "Categoria",
                        options=list(CATEGORIAS_PRODUTO.keys()),
                        required=True
                    ),
                    "tributacao": st.column_config.TextColumn("Tributação", disabled=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Atualizar tributação baseado na categoria
            for idx, row in df_editado.iterrows():
                df_editado.at[idx, 'tributacao'] = CATEGORIAS_PRODUTO[row['categoria']]
            
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("📧 Enviar para Faturamento", type="primary", use_container_width=True):
                    if all([st.session_state.config['email_faturamento'], 
                           st.session_state.config['email_remetente'],
                           st.session_state.config['senha_email']]):
                        
                        with st.spinner("Enviando email..."):
                            if enviar_email_faturamento(cliente_selecionado, df_editado, st.session_state.config):
                                # Adicionar à lista de pedidos
                                pedido = {
                                    'id': len(st.session_state.pedidos) + 1,
                                    'cliente': cliente_selecionado,
                                    'data': datetime.now().strftime('%d/%m/%Y %H:%M'),
                                    'itens': len(df_editado),
                                    'quantidade_total': int(df_editado['quantidade'].sum()),
                                    'status': 'Enviado para Faturamento',
                                    'dados': df_editado.to_dict('records')
                                }
                                st.session_state.pedidos.append(pedido)
                                salvar_pedidos(st.session_state.pedidos)
                                
                                # Limpar dados temporários se existirem
                                if 'temp_df' in st.session_state:
                                    del st.session_state.temp_df
                                if 'temp_email_info' in st.session_state:
                                    del st.session_state.temp_email_info
                                
                                st.success("✅ Email enviado com sucesso!")
                                st.balloons()
                    else:
                        st.error("❌ Configure os emails na barra lateral primeiro!")
            
            with col_btn2:
                # Botão para download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_editado.to_excel(writer, index=False, sheet_name='Pedido')
                output.seek(0)
                
                st.download_button(
                    label="📥 Baixar Planilha",
                    data=output,
                    file_name=f"pedido_{cliente_selecionado.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# TAB 3 - ACOMPANHAMENTO
with tab3:
    st.header("Acompanhamento de Pedidos")
    
    if len(st.session_state.pedidos) == 0:
        st.info("📭 Nenhum pedido registrado ainda.")
    else:
        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_cliente = st.multiselect(
                "Filtrar por Cliente",
                options=list(st.session_state.clientes.keys()),
                default=list(st.session_state.clientes.keys())
            )
        with col_f2:
            filtro_status = st.multiselect(
                "Filtrar por Status",
                options=['Enviado para Faturamento', 'Em Processamento', 'Aguardando Retorno', 'Concluído'],
                default=['Enviado para Faturamento', 'Em Processamento', 'Aguardando Retorno']
            )
        
        # Mostrar pedidos
        for pedido in reversed(st.session_state.pedidos):
            if pedido['cliente'] in filtro_cliente and pedido['status'] in filtro_status:
                with st.expander(f"**Pedido #{pedido['id']}** - {pedido['cliente']} - {pedido['data']} - Status: {pedido['status']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total de Itens", pedido['itens'])
                    col2.metric("Quantidade Total", pedido['quantidade_total'])
                    col3.metric("Status", pedido['status'])
                    
                    st.dataframe(pd.DataFrame(pedido['dados']), use_container_width=True)
                    
                    # Atualizar status
                    novo_status = st.selectbox(
                        "Atualizar Status",
                        options=['Enviado para Faturamento', 'Em Processamento', 'Aguardando Retorno', 'Concluído'],
                        index=['Enviado para Faturamento', 'Em Processamento', 'Aguardando Retorno', 'Concluído'].index(pedido['status']),
                        key=f"status_{pedido['id']}"
                    )
                    
                    if st.button(f"💾 Salvar Status", key=f"btn_status_{pedido['id']}"):
                        pedido['status'] = novo_status
                        salvar_pedidos(st.session_state.pedidos)
                        st.success("Status atualizado!")
                        st.rerun()

# TAB 4 - FINALIZAR PEDIDO
with tab4:
    st.header("Finalizar e Retornar ao Cliente")
    
    pedidos_pendentes = [p for p in st.session_state.pedidos if p['status'] != 'Concluído']
    
    if len(pedidos_pendentes) == 0:
        st.info("📭 Nenhum pedido pendente de finalização.")
    else:
        pedido_selecionado = st.selectbox(
            "Selecione o Pedido",
            options=range(len(pedidos_pendentes)),
            format_func=lambda x: f"Pedido #{pedidos_pendentes[x]['id']} - {pedidos_pendentes[x]['cliente']} - {pedidos_pendentes[x]['data']}"
        )
        
        pedido = pedidos_pendentes[pedido_selecionado]
        
        st.subheader(f"Pedido #{pedido['id']} - {pedido['cliente']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nota_fiscal = st.text_input("Número da Nota Fiscal", placeholder="Ex: 12345")
        
        with col2:
            st.metric("Total de Itens", pedido['itens'])
        
        st.subheader("Itens Não Atendidos")
        st.caption("Deixe em branco se todos os itens foram atendidos")
        
        df_pedido = pd.DataFrame(pedido['dados'])
        
        itens_nao_atendidos = []
        for idx, row in df_pedido.iterrows():
            col_check, col_info = st.columns([1, 4])
            with col_check:
                nao_atendido = st.checkbox(
                    "Não atendido",
                    key=f"check_{pedido['id']}_{idx}"
                )
            with col_info:
                if nao_atendido:
                    st.write(f"**{row['codigo']}** - {row['nome']} (Qtd: {row['quantidade']})")
                    motivo = st.text_input(
                        "Motivo",
                        key=f"motivo_{pedido['id']}_{idx}",
                        placeholder="Ex: Fora de estoque"
                    )
                    if motivo:
                        itens_nao_atendidos.append({
                            'codigo': row['codigo'],
                            'nome': row['nome'],
                            'quantidade': row['quantidade'],
                            'motivo': motivo
                        })
        
        st.markdown("---")
        
        if st.button("📧 Enviar Retorno ao Cliente", type="primary", use_container_width=True):
            if nota_fiscal:
                if all([st.session_state.config['email_remetente'], st.session_state.config['senha_email']]):
                    with st.spinner("Enviando retorno..."):
                        if enviar_retorno_cliente(pedido['cliente'], nota_fiscal, itens_nao_atendidos, st.session_state.config):
                            # Encontrar o pedido original na lista e atualizar
                            for p in st.session_state.pedidos:
                                if p['id'] == pedido['id']:
                                    p['status'] = 'Concluído'
                                    p['nota_fiscal'] = nota_fiscal
                                    p['itens_nao_atendidos'] = itens_nao_atendidos
                                    break
                            
                            salvar_pedidos(st.session_state.pedidos)
                            st.success("✅ Retorno enviado com sucesso!")
                            st.balloons()
                            st.rerun()
                else:
                    st.error("❌ Configure os emails na barra lateral primeiro!")
            else:
                st.error("❌ Informe o número da Nota Fiscal!")

# Footer
st.markdown("---")
st.caption("Sistema de Gerenciamento de Pedidos - Livraria © 2025")