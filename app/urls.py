
from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('',views.index, name='index'),
    path('register',views.register, name='register'),
    path('log_in',views.log_in, name='log_in'),
    path('dashboard',views.dashboard, name='dashboard'),
    path('projects',views.projects, name='projects'),
    path('cards/<str:model_name>',views.cards, name='cards'),
    path('videocards/<str:model_name>',views.videocards, name='videocards'),
    path('posecards/<str:model_name>',views.posecards, name='posecards'),
    path('csv_train',views.csv_train, name='csv_train'),
    path('csv_test',views.csv_test, name='csv_test'),
    path('log_out',views.log_out, name='log_out'),
    path('train_model/', views.train_model, name='train_model'),
]
