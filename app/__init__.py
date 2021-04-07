from flask import Flask, make_response, render_template, request
from .func import check_email, clean_files, send_email, create_file, save_file
from .rna import calculate


app = Flask(__name__)
project = 'NCYPred'


# VIEW - PÁGINA SOBRE O PROJETO
@app.route('/about')
def about():
	page, title = 'about', 'About'
	return render_template('about.html', page=page, project=project, title=title)


# VIEW - PÁGINA DE ENVIO (PÁGINA INICIAL)
# ACTION - PROCESSA O ARQUIVO OU TEXTO FORNECIDO E RETORNA UM JSON COM OS RESULTADOS
@app.route('/')
@app.route('/index')
@app.route('/submit', methods=['GET', 'POST'])
def submit():
	page, title = 'submit', 'Submit'

	if request.method == 'GET':
		return render_template('submit.html', page=page, project=project, title=title)

	elif request.method == 'POST':
		data = function()
		return render_template('submit.html', page=page, project=project, title=title, message=data['message'], status=data['status'], data=data['data'])


# VIEW - PÁGINA DE AJUDA
@app.route('/help')
def help():
	page, title = 'help', 'Help'
	return render_template('help.html', page=page, project=project, title=title)


# VIEW - PÁGINA DE EXPLICAÇÃO DO ALGORITMO
@app.route('/algorithm')
def algorithm():
	page, title = 'algorithm', 'Algorithm'
	return render_template('algorithm.html', page=page, project=project, title=title)


# VIEW - PÁGINA DO TIME
@app.route('/team')
def team():
	page, title = 'team', 'Team'
	return render_template('team.html', page=page, project=project, title=title)


# VIEW - PÁGINA DE CONTATO
@app.route('/contact')
def contact():
	page, title = 'contact', 'Contact'
	return render_template('contact.html', page=page, project=project, title=title)


# ACTION - PROCESSA O ARQUIVO OU TEXTO FORNECIDO E RETORNA UM JSON COM OS RESULTADOS
@app.route('/process', methods=['POST'])
def process():
	if request.method == 'POST':
		data = function()

	response = make_response(data)
	response.headers['Content-Type'] = 'application/json'
	return response


# FUNCTION - PROCESSA O ARQUIVO OU TEXTO FORNECIDO
def function():
	data = {'data': ''}

	if 'file' in request.files and request.files['file'].filename != '': # ENVIOU UM ARQUIVO (IGNORA CASO TENHA ENVIADO UMA SEQUÊNCIA DE TEXTO)
		file = request.files['file']
		path = save_file(file)
		status = bool(path)

		try: # TENTA CALCULAR A SEQUÊNCIA INFORMADA
			path = calculate(path)
			data['message'] = 'The provided file has been processed. ' # O ARQUIVO FORNECIDO FOI PROCESSADO
			status = True

		except: # A SEQUÊNCIA DE RNA INFORMADA POSSUI ALGUM ERRO
			data['message'] = 'Unable to process, there is an error in the file provided.' # NÃO FOI POSSÍVEL PROCESSAR, HÁ UM ERRO NO ARQUIVO FORNECIDO
			status = False

	elif 'sequence' in request.form and len(request.form['sequence']) > 0: # ENVIOU UMA SEQUÊNCIA DE TEXTO
		sequence = request.form['sequence'].strip()

		try: # TENTA CALCULAR A SEQUÊNCIA INFORMADA
			input = create_file(sequence)
			path = calculate(input)
			data['message'] = 'The reported string has been processed. ' # A SEQUÊNCIA INFORMADA FOI PROCESSADA
			status = True

		except: # A SEQUÊNCIA DE RNA INFORMADA POSSUI ALGUM ERRO
			data['message'] = 'Could not process, there is an error in the reported string.' # NÃO FOI POSSÍVEL PROCESSAR, HÁ UM ERRO NA SEQUÊNCIA INFORMADA
			status = False

	else: # NÃO ENVIOU UMA SEQUÊNCIA DE TEXTO
		data['message'] = 'Unable to process, no RNA sequences were reported.' # NÃO FOI POSSÍVEL PROCESSAR, NENHUMA SEQUÊNCIA DE RNA FOI INFORMADA
		status = False

	if status and 'email' in request.form and len(request.form['email']) > 0: # INFORMOU UM E-MAIL
		email = request.form['email'].strip()

		if check_email(email): # O E-MAIL INFORMADO É VÁLIDO
			if send_email(email, path): # O SERVIDOR CONSEGUIU ENVIAR O E-MAIL
				data['message'] += 'An email with the results has been sent to your message box.' # UM E-MAIL COM OS RESULTADOS FOI ENVIADO PARA SUA CAIXA DE MENSAGENS

			else: # HOUVE UM PROBLEMA NO SERVIDOR AO ENVIAR O E-MAIL
				data['message'] += 'It was not possible to send the email, please download the result file.' # NÃO FOI POSSÍVEL ENVIAR O E-MAIL, FAÇA O DOWNLOAD DO ARQUIVO DE RESULTADO
				data['data'] = '<a href="' + path[3:] + '">Click here to download.</a>' # CLIQUE AQUI PARA FAZER O DOWNLOAD

		else: # O E-MAIL INFORMADO É INVÁLIDO
			data['message'] += 'O e-mail fornecido é inválido, faça o download do arquivo de resultado.'
			data['data'] = '<a href="' + path[3:] + '">Click here to download.</a>' # CLIQUE AQUI PARA FAZER O DOWNLOAD

	elif status: # NÃO INFORMOU UM E-MAIL
		data['message'] += 'Foi gerado um link que irá durar 3 dias.'
		data['data'] = '<a href="' + path[3:] + '">Click here to download.</a>' # CLIQUE AQUI PARA FAZER O DOWNLOAD

	data['status'] = status
	return data


# VIEW - PÁGINA DE LIMPEZA DE ARQUIVOS ANTIGOS
@app.route('/clean')
def clean():
	clean_files(3)
	return ''

# RUN - FUNÇÃO DE START
if __name__ == '__main__':
    app.run(host='0.0.0.0')