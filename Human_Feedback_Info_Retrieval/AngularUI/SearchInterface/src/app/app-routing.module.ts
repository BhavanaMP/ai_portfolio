import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { LoginComponent } from './login/login.component';
import { ResultsUiComponent } from './results-ui/results-ui.component';
import { SearchUIComponent } from './search-ui/search-ui.component';

const routes: Routes = [
  {
    path: '', 
    component : LoginComponent, 

  }, 
  {
    path: 'home', 
    component : LoginComponent, 

  },
  {
    path : 'search',
     component : SearchUIComponent, 
    },
    {
      path : 'results',
       component : ResultsUiComponent, 
      },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {
 
 }
