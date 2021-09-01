from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from django.conf import settings
from django.shortcuts import get_object_or_404, redirect



class IndexPageView(TemplateView):
    template_name = 'main/index.html'

class ChangeLanguageView(TemplateView):
    template_name = 'main/change_language.html'


# class HomePageView(LoginRequiredMixin ,TemplateView):
#     def dispatch(self, request):
#         name = 'layouts/halfmoon.html'
#         # Sets a test cookie to make sure the user has cookies enabled
#         if request.user.is_authenticated:
#             return redirect(settings.LOGIN_REDIRECT_URL)

#         return redirect(name)
