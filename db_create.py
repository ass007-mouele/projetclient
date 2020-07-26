from app import Post, db


db.create_all()

#p=Post(Date="2018-08-07",Heure=" 06:00",Bassin="GB",Transparence="Clair",Temperature_de_l_eau=27,pH=7.5,DPD_1=1.7,DPD_3="0.4",combine=1.9,libre_actif=0.8,compteur="45678")
#db.session.add(p)

db.session.commit()
