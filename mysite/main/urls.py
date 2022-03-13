"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from .import views

app_name = 'main'

urlpatterns = [
    

    path("obj_detection",views.obj_detection, name="obj_detection"),
    path("face_recognition",views.face_recognition, name="face_recognition"),
    path("music_generation",views.music_generation, name="music_generation"),
    path("face_generation",views.face_generation, name="face_generation"),
    path("", views.First_page, name = "First_page"),
    path("homepage", views.homepage, name = "homepage"),
    path("register", views.register, name="register"),
    path("logout",views.logout_req,name = "logout"),
    path("login",views.login_req,name = "login_req"),
    path("Extras",views.extras,name = 'extras'),
    path("object",views.index, name="index"),
    path("collect",views.collect, name="collect"),
    path("run_face",views.run_face, name="run_face"),
    path("updating_weights_for_face_recognition",views.updating_weights_for_face_recognition, name="updating_weights_for_face_recognition"),
    path("gen_music",views.gen_music, name="gen_music"),    
    path("playAudioFile",views.playAudioFile, name="playAudioFile"),  
    path("gen_img",views.gen_img,name="gen_img"),   
    path("download_img",views.download_img,name="download_img"),
    path("First_page",views.First_page,name="First_page"),
    path("projects",views.projects,name="projects"),
    path("blog",views.blog,name="blog"),
    path("videos",views.videos,name="videos"),
    path("notion",views.notion,name="notion"),

]
