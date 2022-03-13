# Generated by Django 2.2.2 on 2019-09-26 18:13

import ckeditor.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Definitions',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('definition', ckeditor.fields.RichTextField(blank=True, null=True)),
            ],
        ),
        migrations.AddField(
            model_name='tutorial',
            name='tutorial_color',
            field=models.CharField(default='#FF0000', max_length=10, verbose_name='color'),
        ),
        migrations.AlterField(
            model_name='tutorial',
            name='tutorial_content',
            field=models.TextField(verbose_name='content'),
        ),
        migrations.AlterField(
            model_name='tutorial',
            name='tutorial_title',
            field=models.CharField(max_length=200, verbose_name='title'),
        ),
    ]
