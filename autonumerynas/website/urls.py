from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.home, name='home'),
    path('parduoti-numerius/', views.sell_license_plates, name='sell_license_plates'),
    path('prisijungti/', auth_views.LoginView.as_view(), name='login'),
    path('atsijungti/', auth_views.LogoutView.as_view(), name='logout'),
    path('registracija/', views.register, name='register'),
    # path('pirkti-numerius', views.buy_license_plates, name='buy_license_plates'),
    # path('ivertinti-numerius', views.evaluate_plate, name="evaluate_plate"),
    # path('duk', views.duk, name="duk"),
    # path('atsiliepimai', views.atsiliepimai, name="atsiliepimai"),
    # path('susisiekime', views.susisiekime, name="susisiekime"),
    # path('apie-mus', views.apie_mus, name="apie_mus"),
    # path('slapukai', views.slapukai, name="slapukai"),
    # path('apie-mus', views.apie_mus, name="apie_mus"),
    # path('naudojimosi-taisykles', views.naudojimosi_taisykles, name="naudojimosi_taisykles"),
]
