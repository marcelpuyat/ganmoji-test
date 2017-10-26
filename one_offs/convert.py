import commands


s = commands.getstatusoutput('ls data')
for filename in s[1].split():
	status = commands.getstatusoutput('sips -Z 32 data/' + filename)
	if status[0] == 0:
		print('Successfully converted file: ' + filename)
	else:
		print('Unsuccessful conversion of file ' + filename+'. Status: ' + str(status))
