from django.db import models
from rgbfield.fields import RGBColorField
from ckeditor.fields import RichTextField

# Create your models here.
class Tutorial(models.Model): # Thos wil create a table in the sql database and the we need to create coluoms under the class
    tutorial_title = models.CharField(max_length = 200,verbose_name="title")
    tutorial_content = models.TextField(verbose_name="content")
    tutorial_color = models.CharField(max_length=10,default="#FF0000",verbose_name="color")##RGBColorField(default='#fff') 
    
    def __str__(self):
        return self.tutorial_title

class Definitions(models.Model):
	name = models.CharField(max_length = 200)
	definition = RichTextField(blank=True, null = True)

