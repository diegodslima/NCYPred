import os
from app import app as application

environ = 'dev'

if __name__ == '__main__':
	if environ == 'dev':
		application.run(debug=True, use_reloader=True)

	else:
		port = int(os.environ.get('PORT', 5000))
		application.run(host='0.0.0.0', port=port)

# uwsgi --socket 0.0.0.0:5000 --protocol=http -w wsgi:application