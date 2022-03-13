from django.contrib import admin
from .models import Tutorial,Definitions
from tinymce.widgets import TinyMCE
from django.db import models

# Register your models here.

class TutorialAdmin(admin.ModelAdmin):
    fieldsets = [
            ("Title",{'fields':['tutorial_title']}),
            ('content',{'fields':['tutorial_content']}),
            ('color',{'fields':['tutorial_color']}),
             ]
    
    formfield_overrides = {
            models.TextField:{'widget':TinyMCE()}
            }

class Definition(admin.ModelAdmin):
  fieldsets = [
    ("name",{'fields':['name']}),
    ("definition",{'fields':['definition']}),  
  ]

admin.site.register(Definitions,Definition)    
admin.site.register(Tutorial,TutorialAdmin)