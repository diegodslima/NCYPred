from datetime import datetime, timedelta
import os, string, random, re

import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import base64


# ATIVAR O ENVIO DE E-MAILS DO GOOGLE PELO SITE https://www.google.com/settings/security/lesssecureapps
email_address = 'youremail@yourboxmail.com'
email_password = 'yourpassword'


# VERIFICA SE O E-MAIL É VÁLIDO
def check_email(email):
	regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
	return re.search(regex, email)


# ENVIA UMA MENSAGEM PARA O E-MAIL
def send_email(receiver, path):
	message = MIMEMultipart()
	message['Subject'] = 'NCYPred - Result of prediction'
	message['From'] = email_address
	message['To'] = receiver

	text = '''
Hello, the prediction results are ready, just download the file attached to this email.

Thank you very much for using our platform!

NCYPred Team.''' # VERSÃO EM TEXTO SIMPLES

	html = '''
<p>Hello, the prediction results are ready, just download the file attached to this email.</p>
<p>Thank you very much for using our platform!</p>
<p>NCYPred Team.</p>''' # VERSÃO EM TEXTO HTML

	# TRANSFORMA O TEXTO E O HTML EM MIMETEXT SIMPLES
	part = MIMEText(text, 'plain') # part = MIMEText(html, 'html')

	# ADICIONA O TEXTO E O HTML À MENSAGEM MIMEMULTIPART
	message.attach(part)

	# ABRE O ARQUIVO NO MODO BINÁRIO E CODIFICA O ANEXO
	file = open(path, 'rb')
	payload = MIMEBase('application', 'octate-stream', Name=os.path.basename(path))
	payload.set_payload((file).read())
	encoders.encode_base64(payload)

	# ADICIONA AO CABEÇALHO O ARQUIVO
	payload.add_header('Content-Decomposition', 'attachment', filename=os.path.basename(path))
	message.attach(payload)

	# CRIA UMA CONEXÃO SEGURA COM O SERVIDOR E ENVIA O E-MAIL
	context = ssl.create_default_context()
	with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server: # with smtplib.SMTP('smtp.gmail.com: 587') as server:
		try:
			server.ehlo() # server.starttls()
			server.login(email_address, email_password)
			server.sendmail(email_address, receiver, message.as_string())
			server.quit()
			return True
		except:
			return False


# SALVA O ARQUIVO NA PASTA ESTÁTICA DE ENTRADA
def save_file(file):
	root = os.path.abspath('.')
	name = datetime.now().strftime('%Y%m%d%H%M%S') + ''.join(random.choice(string.ascii_letters) for i in range(6)) + file.filename
	path = os.path.join(root, 'app', 'static', 'input', name)

	file.save(path)
	if os.path.isfile(path):
		return os.path.splitext(name)[0]
	return ''


# SALVA O TEXTO EM UM ARQUIVO NA PASTA ESTÁTICA DE ENTRADA
def create_file(text):
	root = os.path.abspath('.')
	name = datetime.now().strftime('%Y%m%d%H%M%S') + ''.join(random.choice(string.ascii_letters) for i in range(6)) + '.fasta'
	path = os.path.join(root, 'app', 'static', 'input', name)

	file = open(path, 'w')
	rows = file.write(text)
	file.close()

	if rows:
		return os.path.splitext(name)[0]
	return ''


# REMOVE OS ARQUIVOS QUE ESTÃO HÁ MUITOS DIAS NO SERVIDOR
def clean_files(days):
	root = os.path.abspath('.')
	input_path = os.path.join(root, 'app', 'static', 'input', '')
	output_path = os.path.join(root, 'app', 'static', 'output', '')

	input_dir = os.listdir('./app/static/input')
	output_dir = os.listdir('./app/static/output')

	current_date = datetime.now() - timedelta(days=days)

	for i in input_dir:
		day = datetime(int(i[:4]), int(i[4:6]), int(i[6:8]))
		if day <= current_date:
			os.remove(input_path + i)

	for o in output_dir:
		day = datetime(int(o[:4]), int(o[4:6]), int(o[6:8]))
		if day <= current_date:
			os.remove(output_path + o)