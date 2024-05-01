initdb:
	python migrate.py

run:
	flask --app main run --host=0.0.0.0

start-runner:
	python runner.py

req:
	curl -X POST -H 'Content-Type: application/json' localhost:5000/submissions \
		-d '{"code": "aW1wb3J0IHRpbWUKCnByaW50KCJ0aGlzIGlzIGZyb20gYW5vdGhlci5weSIsIGZsdXNoPVRydWUpCgp0cnk6CiAgICB3aXRoIG9wZW4oImhvaG8udHh0IiwgJ3crJykgYXMgZjoKICAgICAgICBmLndyaXRlKCJha2Nta2FjYSIpCiAgICAgICAgZi5mbHVzaCgpCmV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgIHByaW50KGUpCgojIHdoaWxlIFRydWU6CiMgICAgIHByaW50KCJzdGFydCBzbGVlcGluZyIsIGZsdXNoPVRydWUpCiMgICAgIHRpbWUuc2xlZXAoMSk="}'