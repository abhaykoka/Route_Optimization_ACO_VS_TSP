from django.urls import path
from . import views

urlpatterns = [
    path('', views.input_view, name='input'),
    path('project_info/', views.project_info, name='project_info'),
    path('output/', views.output_view, name='output'),
]
