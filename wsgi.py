from _init_ import app , db
import psycopg2

if __name__=='__main__':
	db.create_all()
	app.run()
  #app.run(debug=True,port=3000)

