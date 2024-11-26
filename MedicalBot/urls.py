from django.urls import path
from . import views
from django.views.generic import TemplateView
from .views import signup, login_view, home, inference
urlpatterns = [
    path('', TemplateView.as_view(template_name="index.html")),
    path('signup/', signup, name='signup'),
    path('login/', login_view, name='login'),
    path('home/', home, name='home'),
    path('inference/', inference, name='inference'),
]
