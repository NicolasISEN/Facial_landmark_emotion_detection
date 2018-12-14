import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def sendAnEMail(subject,message):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()
	server.ehlo()
	server.login("jerome.desoutter.senddata@gmail.com", "ProjetISEN2018")
	
	fromaddr = "jerome.desoutter.senddata@gmail.com"
	toaddr = "jerome.desoutter@isen.yncrea.fr"
	ccaddr = "nicolas.thuylie@orange.fr"
	
	msg = MIMEMultipart()
	msg['From'] = fromaddr
	msg['To'] = toaddr
	msg['Cc'] = ccaddr
	msg['Subject'] = subject
	
	body = message
	msg.attach(MIMEText(body, 'plain'))
	
	txt = msg.as_string()
	server.sendmail(fromaddr,toaddr, txt)
	server.sendmail(fromaddr,ccaddr, txt)
	server.quit()

def sendInfos(numero_model,desc_model,mean_loss,mean_accuracy,landmarks_loss,landmarks_accuracy,expression_loss,expression_accuracy,
	mean_loss_v,mean_accuracy_v,landmarks_loss_v,landmarks_accuracy_v,expression_loss_v,expression_accuracy_v):
	subj = "Modèle numéro " + str(numero_model)
	message = "<Rapport IA Projet ISEN 2018>\n\n"
	message += "Modèle numéro " + str(numero_model) + '\n\n'
	message += "Description\n"
	
	try:
		message += "Batch size : " + str(desc_model["bs"]) + '\n'
		message += "Learning rate : " + str(desc_model["lr"]) + '\n'
		message += "Facteur : " + str(desc_model["f"]) + '\n'
		message += "Patience : " + str(desc_model["p"]) + '\n'
		message += "Dropout : " + str(desc_model["d"]) + '\n'
		message += "Poids : " + str(desc_model["weights"][0]) + " - " + str(desc_model["weights"][1]) + '\n'
		message += "Epoch : " + str(desc_model["epoch"]) + " / " + str(desc_model["nbepochs"]) + '\n\n'
	except Exception as e:
		message += "Failed"
	else:
		pass
	finally:
		pass

	message += "Result\n"
	message += "Mean loss training : " + str(mean_loss) + '\n'
	message += "Mean accuracy training : " + str(mean_accuracy) + '\n'
	message += "Landmarks loss training : " + str(landmarks_loss) + '\n'
	message += "Landmarks accuracy training : " + str(landmarks_accuracy) + '\n'
	message += "Expression loss training : " + str(expression_loss) + '\n'
	message += "Expression accuracy training : " + str(expression_accuracy) + '\n'

	message += "Mean loss validation: " + str(mean_loss_v) + '\n'
	message += "Mean accuracy validation : " + str(mean_accuracy_v) + '\n'
	message += "Landmarks loss validation : " + str(landmarks_loss_v) + '\n'
	message += "Landmarks accuracy validation : " + str(landmarks_accuracy_v) + '\n'
	message += "Expression loss validation : " + str(expression_loss_v) + '\n'
	message += "Expression accuracy validation : " + str(expression_accuracy_v) + '\n'

	sendAnEMail(subj,message)

#desc = dict()
#desc["bs"] = 8
#desc["lr"] = 0.0001
#desc["f"] = 0.2
#desc["p"] = 5
#desc["d"] = 0.2
#desc["weights"] = [1,0.0001]
#desc["epoch"] = 1
#desc["nbepochs"] = 30

def help():
	print("Example : sendInfos(1,desc,1.5,0.5,0.0015,0.4,1.6,0.6,1.5,0.5,0.0015,0.4,1.6,0.6)")