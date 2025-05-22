from setuptools import find_packages, setup

setup(
    name="E-commerce_rag_chatbot",
    version="0.0.1",
    author="Gopichand",
    author_email="gopichand.datascientist9977@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain ','langchain-google-genai','datasets','pypdf','python-dotenv','flask']
)