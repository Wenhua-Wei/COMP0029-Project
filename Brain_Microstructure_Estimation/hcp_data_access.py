from dmipy.hcp_interface import downloader_aws

'''
Method to dowload HCP data using Dmipy library
'''

# AWS public and secret keys
public_aws_key = 'AKIAXO65CT57JKK7BT2I'
secret_aws_key = 'B8rAqQLKud4jjarcfCAorjuACPdNZljPy97Y51F7'

hcp_interface = downloader_aws.HCPInterface(
    your_aws_public_key=public_aws_key,
    your_aws_secret_key=secret_aws_key)

available_hcp_subjects = hcp_interface.available_subjects
print(available_hcp_subjects[5])

hcp_interface.download_subject(subject_ID=available_hcp_subjects[5])